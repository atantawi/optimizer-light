package config

// TODO: add json validation and default values

// All data related to the system (accelerators, models, service classes, ...)
type SystemData struct {
	Spec SystemSpec `json:"system"`
}

// Specifications for system data
type SystemSpec struct {
	// static data
	Accelerators   AcceleratorData  `json:"acceleratorData"`  // accelerator data
	Models         ModelData        `json:"modelData"`        // model data
	ServiceClasses ServiceClassData `json:"serviceClassData"` // service class data
	Servers        ServerData       `json:"serverData"`       // server data
	Optimizer      OptimizerData    `json:"optimizerData"`    // optimizer data

	// dynamic data
	Capacity CapacityData `json:"capacityData"` // data about accelerator type availability
}

// Data related to an Accelerator
type AcceleratorData struct {
	Spec []AcceleratorSpec `json:"accelerators"` // accelerator specs
}

// Specifications for accelerator data
type AcceleratorSpec struct {
	Name         string    `json:"name"`         // name of accelerator
	Type         string    `json:"type"`         // name of accelerator type (e.g. A100)
	Multiplicity int       `json:"multiplicity"` // number of cards of type for this accelerator
	MemSize      int       `json:"memSize"`      // GB
	MemBW        int       `json:"memBW"`        // GB/sec
	Power        PowerSpec `json:"power"`        // power consumption specs
	Cost         float32   `json:"cost"`         // cents/hr
}

// Specifications for Accelerator power consumption data (Watts)
type PowerSpec struct {
	Idle     int     `json:"idle"`     // idle power
	Full     int     `json:"full"`     // full utilization power
	MidPower int     `json:"midPower"` // power at inflection point
	MidUtil  float32 `json:"midUtil"`  // utilization at inflection point
}

// Data about accelerator type availability
type CapacityData struct {
	Count []AcceleratorCount `json:"count"` // count of accelerator types
}

// Count of accelerator types in the system
type AcceleratorCount struct {
	Type  string `json:"type"`  // name of accelerator type
	Count int    `json:"count"` // number of available units
}

// Data related to a Model
type ModelData struct {
	PerfData []ModelAcceleratorPerfData `json:"models"` // performance data for model on accelerators
}

// Specifications for a combination of a model and accelerator data
type ModelAcceleratorPerfData struct {
	Name         string  `json:"name"`         // model name
	Acc          string  `json:"acc"`          // accelerator name
	AccCount     int     `json:"accCount"`     // number of accelerator units used by model
	Alpha        float32 `json:"alpha"`        // alpha parameter of ITL
	Beta         float32 `json:"beta"`         // beta parameter of ITL
	MaxBatchSize int     `json:"maxBatchSize"` // max batch size based on average number of tokens per request
	AtTokens     int     `json:"atTokens"`     // average number of tokens per request assumed in max batch size calculation
}

// Data related to a service class SLOs
type ServiceClassData struct {
	Spec []ServiceClassSpec `json:"serviceClasses"`
}

// Specifications of SLO data for a combination of a service class and a model
type ServiceClassSpec struct {
	Name     string  `json:"name"`     // service class name
	Model    string  `json:"model"`    // model name
	Priority int     `json:"priority"` // (non-negative) priority (lower value is higher priority)
	SLO_ITL  float32 `json:"slo-itl"`  // inter-token latency (msec)
	SLO_TTW  float32 `json:"slo-ttw"`  // request waiting time (msec)
	SLO_TPS  float32 `json:"slo-tps"`  // throughput (tokens/sec)
}

// Data related to a Server
type ServerData struct {
	Spec []ServerSpec `json:"servers"`
}

// Specifications of a server
type ServerSpec struct {
	Name            string         `json:"name"`            // server name
	Class           string         `json:"class"`           // service class name
	Model           string         `json:"model"`           // model name
	KeepAccelerator bool           `json:"keepAccelerator"` // option to not change accelerator
	MinNumReplicas  int            `json:"minNumReplicas"`  // minimum number of replicas
	CurrentAlloc    AllocationData `json:"currentAlloc"`    // current allocation
	DesiredAlloc    AllocationData `json:"desiredAlloc"`    // desired allocation
}

// Specifications of server load statistics
type ServerLoadSpec struct {
	ArrivalRate float32 `json:"arrivalRate"` // req/min
	AvgLength   int     `json:"avgLength"`   // number of tokens
	ArrivalCOV  float32 `json:"arrivalCOV"`  // coefficient of variation of inter-request arrival time
	ServiceCOV  float32 `json:"serviceCOV"`  // coefficient of variation of request service time
}

// Data related to Optimizer
type OptimizerData struct {
	Spec OptimizerSpec `json:"optimizer"`
}

// Specifications for optimizer data
type OptimizerSpec struct {
	Unlimited     bool `json:"unlimited"`     // unlimited number of accelerator types (for capacity planning and/or cloud)
	Heterogeneous bool `json:"heterogeneous"` // heterogeneous accelerators assigned to same inference server
}

type AllocationSolution struct {
	Spec map[string]AllocationData `json:"allocations"` // map of server names to allocation data
}

// Data about a server allocation
type AllocationData struct {
	Accelerator string         `json:"accelerator"` // accelerator name
	NumReplicas int            `json:"numReplicas"` // number of replicas
	MaxBatch    int            `json:"maxBatch"`    // max batch size
	Cost        float32        `json:"cost"`        // cost of allocation
	ITLAverage  float32        `json:"itlAverage"`  // average ITL
	WaitAverage float32        `json:"waitAverage"` // average wait time
	Load        ServerLoadSpec `json:"load"`        // server load statistics
}
