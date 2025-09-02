package core

import (
	"bytes"
	"fmt"
	"math"

	"github.com/llm-inferno/optimizer-light/pkg/analyzer"
	"github.com/llm-inferno/optimizer-light/pkg/config"
)

// Allocation details of an accelerator to a server
type Allocation struct {
	accelerator string  // name of accelerator
	numReplicas int     // number of server replicas
	batchSize   int     // max batch size
	cost        float32 // cost of this allocation
	value       float32 // value of this allocation
	servTime    float32 // expected average token service time
	waitTime    float32 // expected average request queueing time
	rho         float32 // average concurrently running requests / max batch size

	maxArrvRatePerReplica float32 // maximum arrival rate per replica
}

// Create an allocation of an accelerator to a server; nil if not feasible
func CreateAllocation(serverName string, gName string) *Allocation {
	var (
		acc *Accelerator

		server *Server
		load   *config.ServerLoadSpec

		model *Model
		perf  *config.ModelAcceleratorPerfData

		svc    *ServiceClass
		target *Target
	)

	// get accelerator info
	if acc = GetAccelerator(gName); acc == nil {
		return nil
	}

	// get server info
	if server = GetServer(serverName); server == nil {
		return nil
	}
	if load = server.Load(); load == nil || load.ArrivalRate < 0 || load.AvgLength < 0 {
		return nil
	}

	// get model info
	modelName := server.ModelName()
	if model = GetModel(modelName); model == nil {
		return nil
	}
	if perf = model.PerfData(gName); perf == nil {
		return nil
	}

	// get service class info
	if svc = GetServiceClass(server.ServiceClassName()); svc == nil {
		return nil
	}
	if target = svc.ModelTarget(modelName); target == nil {
		return nil
	}

	// handle zero traffic case
	if load.ArrivalRate == 0 || load.AvgLength == 0 {
		return zeroLoadAllocation(server, model, acc, perf)
	}

	// calculate max batch size (N) based on average request length (K)
	K := load.AvgLength

	// use maxBatchSize from configured value or scaled performance data
	var N int
	if server.maxBatchSize > 0 {
		N = server.maxBatchSize
	} else {
		N = max(perf.MaxBatchSize*perf.AtTokens/K, 1)
	}
	maxQueue := N * config.MaxQueueToBatchRatio

	// create queue analyzer
	// TODO: add data for gamma and delta prefill parameters
	qConfig := &analyzer.Configuration{
		MaxBatchSize: N,
		MaxQueueSize: maxQueue,
		ServiceParms: &analyzer.ServiceParms{
			Prefill: &analyzer.PrefillParms{
				Gamma: perf.Alpha,
				Delta: perf.Beta,
			},
			Decode: &analyzer.DecodeParms{
				Alpha: perf.Alpha,
				Beta:  perf.Beta,
			},
		},
	}

	// TODO: add input tokens for prefill
	requestData := &analyzer.RequestSize{
		AvgInputTokens:  1,
		AvgOutputTokens: K,
	}

	queueAnalyzer, err := analyzer.NewQueueAnalyzer(qConfig, requestData)
	if err != nil {
		fmt.Println(err)
		return nil
	}

	waitTimeLimit := target.TTW / config.SLOMargin // distribution of waiting time assumed exponential
	targetPerf := &analyzer.TargetPerf{
		TargetTTFT: waitTimeLimit,
		TargetITL:  target.ITL,
		TargetTPS:  target.TPS,
	}

	// determine max rates to satisfy targets
	_, metrics, _, err := queueAnalyzer.Size(targetPerf)
	if err != nil {
		// fmt.Println(err)
		return nil
	}
	rateStar := metrics.Throughput

	// calculate number of replicas
	var totalRate float32
	if target.TPS == 0 {
		totalRate = load.ArrivalRate / 60
	} else {
		totalRate = target.TPS / float32(K)
	}
	numReplicas := int(math.Ceil(float64(totalRate) / float64(rateStar)))
	numReplicas = max(numReplicas, server.minNumReplicas)

	// calculate cost
	totalNumInstances := model.NumInstances(gName) * numReplicas
	cost := acc.Cost() * float32(totalNumInstances)

	// analyze queue of one replica
	rate := totalRate / float32(numReplicas)
	metrics, err = queueAnalyzer.Analyze(rate)
	if err != nil {
		fmt.Println(err)
		return nil
	}
	rho := metrics.Rho
	servTime := metrics.AvgTokenTime
	wait := metrics.AvgWaitTime
	// fmt.Printf("numReplicas=%d; batchSize=%d; rate=%v, tokenTime=%v; wait=%v; \n", numReplicas, N, rate, servTime, wait)

	alloc := &Allocation{accelerator: gName, numReplicas: numReplicas, batchSize: N,
		cost: cost, servTime: servTime, waitTime: wait, rho: rho, maxArrvRatePerReplica: rateStar / 1000}
	alloc.SetValue(alloc.cost)
	return alloc
}

// Change number of replicas in allocation and re-evaluate performance, assuming total load on a server
func (a *Allocation) AdjustNumReplicas(numReplicas int, server *Server, model *Model) error {
	if a.numReplicas < 1 || a.batchSize < 1 {
		return fmt.Errorf("invalid current numReplicas (%d) or batchSize (%d)", a.numReplicas, a.batchSize)
	}

	// get load statistics
	load := server.Load()
	if load == nil {
		return fmt.Errorf("missing server load spec for server %s", server.name)
	}
	K := load.AvgLength
	totalRate := load.ArrivalRate / 60

	// check if throughtput constrained
	var target *Target
	if svClass := GetServiceClass(server.ServiceClassName()); svClass != nil {
		if target = svClass.ModelTarget(model.name); target != nil {
			if target.TPS > 0 && K > 0 {
				totalRate = target.TPS / float32(K)
			}
		}
	}

	// get performance parameters
	perf := model.PerfData(a.accelerator)
	if perf == nil {
		return fmt.Errorf("missing performance data for model %s on accelerator %s", model.Name(), a.accelerator)
	}

	// calculate queue statistics
	N := a.batchSize
	maxQueue := N * config.MaxQueueToBatchRatio

	// create queue analyzer
	// TODO: add data for gamma and delta prefill parameters
	qConfig := &analyzer.Configuration{
		MaxBatchSize: N,
		MaxQueueSize: maxQueue,
		ServiceParms: &analyzer.ServiceParms{
			Prefill: &analyzer.PrefillParms{
				Gamma: perf.Alpha,
				Delta: perf.Beta,
			},
			Decode: &analyzer.DecodeParms{
				Alpha: perf.Alpha,
				Beta:  perf.Beta,
			},
		},
	}

	// TODO: add input tokens for prefill
	requestData := &analyzer.RequestSize{
		AvgInputTokens:  1,
		AvgOutputTokens: K,
	}

	queueAnalyzer, err := analyzer.NewQueueAnalyzer(qConfig, requestData)
	if err != nil {
		fmt.Println(err)
		return nil
	}

	// analyze queue under load
	rate := totalRate / float32(numReplicas)
	var metrics *analyzer.AnalysisMetrics
	if metrics, err = queueAnalyzer.Analyze(rate); err != nil {
		fmt.Println(err)
		return nil
	}

	// set allocation fields
	a.rho = metrics.Rho
	a.servTime = metrics.AvgTokenTime
	a.waitTime = metrics.AvgWaitTime

	// adjust cost and value
	factor := float32(numReplicas) / float32(a.numReplicas)
	a.cost *= factor
	a.value *= factor

	waitTimeLimit := target.TTW / config.SLOMargin // distribution of waiting time assumed exponential
	targetPerf := &analyzer.TargetPerf{
		TargetTTFT: waitTimeLimit,
		TargetITL:  target.ITL,
		TargetTPS:  target.TPS,
	}

	// determine max rates to satisfy targets
	if _, metrics, _, err = queueAnalyzer.Size(targetPerf); err != nil {
		// fmt.Println(err)
		return nil
	}
	a.maxArrvRatePerReplica = metrics.Throughput / 1000
	a.numReplicas = numReplicas
	return nil
}

func (a *Allocation) Scale(serverName string) (alloc *Allocation, inc int) {
	var (
		acc    *Accelerator
		server *Server
		load   *config.ServerLoadSpec
	)

	// get server info
	if server = GetServer(serverName); server == nil {
		return nil, 0
	}
	if load = server.Load(); load == nil {
		return nil, 0
	}

	// get accelerator info
	gName := a.accelerator
	if acc = GetAccelerator(gName); acc == nil {
		return nil, 0
	}

	// create new allocation
	alloc = CreateAllocation(serverName, gName)
	inc = alloc.numReplicas - a.numReplicas
	return alloc, inc
}

func (a *Allocation) ReAllocate(serverName string) (*Allocation, string) {
	minVal := float32(0)
	var minAlloc *Allocation
	for gName := range GetAccelerators() {
		if alloc := CreateAllocation(serverName, gName); alloc != nil {
			if minVal == 0 || alloc.value < minVal {
				minVal = alloc.value
				minAlloc = alloc
			}
		}
	}
	if minAlloc == nil {
		return nil, ""
	}
	return minAlloc, minAlloc.accelerator
}

func (a *Allocation) Accelerator() string {
	return a.accelerator
}

func (a *Allocation) NumReplicas() int {
	return a.numReplicas
}

func (a *Allocation) SetNumReplicas(n int) {
	a.numReplicas = n
}

func (a *Allocation) MaxBatchSize() int {
	return a.batchSize
}

func (a *Allocation) SetMaxBatchSize(batchSize int) {
	a.batchSize = batchSize
}

func (a *Allocation) MaxArrvRatePerReplica() float32 {
	return a.maxArrvRatePerReplica
}

func (a *Allocation) MaxRPM() float32 {
	return a.maxArrvRatePerReplica * 1000 * 60
}

func (a *Allocation) Cost() float32 {
	return a.cost
}

func (a *Allocation) SetCost(cost float32) {
	a.cost = cost
}

func (a *Allocation) Value() float32 {
	return a.value
}

// Set the value for this allocation (may depend on cost, performance, ...)
func (a *Allocation) SetValue(value float32) {
	a.value = value
}

func (a *Allocation) Saturated(totalRate float32) bool {
	return totalRate > float32(a.numReplicas)*a.MaxRPM()
}

// Allocation in case of zeroload
func zeroLoadAllocation(server *Server, model *Model, acc *Accelerator, perf *config.ModelAcceleratorPerfData) *Allocation {
	maxBatchSize := perf.MaxBatchSize
	if server.maxBatchSize > 0 {
		maxBatchSize = server.maxBatchSize
	}
	numReplicas := server.minNumReplicas
	gName := acc.Name()
	totalNumInstances := model.NumInstances(gName) * numReplicas
	cost := acc.Cost() * float32(totalNumInstances)
	servTime := perf.Alpha + perf.Beta
	minServTime := perf.Alpha + perf.Beta*float32(maxBatchSize)
	maxArrvRatePerReplica := float32(maxBatchSize) / minServTime

	alloc := &Allocation{accelerator: gName, numReplicas: numReplicas, batchSize: maxBatchSize,
		cost: cost, servTime: servTime, waitTime: 0, rho: 0, maxArrvRatePerReplica: maxArrvRatePerReplica}
	alloc.SetValue(alloc.cost)
	return alloc
}

// Calculate penalty for transitioning from this allocation (a) to another allocation (b)
func (a *Allocation) TransitionPenalty(b *Allocation) float32 {
	if a.accelerator == b.accelerator {
		if a.numReplicas == b.numReplicas {
			return 0
		} else {
			return b.cost - a.cost
		}
	}
	return config.AccelPenaltyFactor*(a.cost+b.cost) + (b.cost - a.cost)
}

func (a *Allocation) Clone() *Allocation {
	return &Allocation{
		accelerator: a.accelerator,
		numReplicas: a.numReplicas,
		batchSize:   a.batchSize,
		cost:        a.cost,
		value:       a.value,
		servTime:    a.servTime,
		waitTime:    a.waitTime,
		rho:         a.rho,

		maxArrvRatePerReplica: a.maxArrvRatePerReplica,
	}
}

func (a *Allocation) AllocationData() *config.AllocationData {
	return &config.AllocationData{
		Accelerator: a.accelerator,
		NumReplicas: a.numReplicas,
		MaxBatch:    a.batchSize,
		Cost:        a.cost,
		ITLAverage:  a.servTime,
		WaitAverage: a.waitTime,
	}
}

func AllocationFromData(data *config.AllocationData) *Allocation {
	return &Allocation{
		accelerator: data.Accelerator,
		numReplicas: data.NumReplicas,
		batchSize:   data.MaxBatch,
		cost:        data.Cost,
		servTime:    data.ITLAverage,
		waitTime:    data.WaitAverage,
	}
}

func (a *Allocation) String() string {
	return fmt.Sprintf("{acc=%s; num=%d; maxBatch=%d; cost=%v, val=%v, servTime=%v, waitTime=%v, rho=%v, maxRPM=%v}",
		a.accelerator, a.numReplicas, a.batchSize, a.cost, a.value, a.servTime, a.waitTime, a.rho, a.MaxRPM())
}

// Orchestration difference between two allocations
type AllocationDiff struct {
	oldAccelerator string
	newAccelerator string
	oldNumReplicas int
	newNumReplicas int
	costDiff       float32
}

func CreateAllocationDiff(a *Allocation, b *Allocation) *AllocationDiff {
	if a == nil && b == nil {
		return nil
	}
	oldAccelerator := "none"
	newAccelerator := "none"
	oldNumReplicas := 0
	newNumReplicas := 0
	oldCost := float32(0)
	newCost := float32(0)
	if a != nil {
		oldAccelerator = a.accelerator
		oldNumReplicas = a.numReplicas
		oldCost = a.cost
	}
	if b != nil {
		newAccelerator = b.accelerator
		newNumReplicas = b.numReplicas
		newCost = b.cost
	}
	return &AllocationDiff{
		oldAccelerator: oldAccelerator,
		newAccelerator: newAccelerator,
		oldNumReplicas: oldNumReplicas,
		newNumReplicas: newNumReplicas,
		costDiff:       newCost - oldCost,
	}
}

func (d *AllocationDiff) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "{ %s -> %s, %d -> %d, %v }",
		d.oldAccelerator, d.newAccelerator, d.oldNumReplicas, d.newNumReplicas, d.costDiff)
	return b.String()
}
