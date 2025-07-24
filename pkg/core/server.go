package core

import (
	"fmt"

	"github.com/llm-inferno/optimizer-light/pkg/config"
)

// A server for a service class and model
type Server struct {
	name             string
	serviceClassName string
	modelName        string

	// server load statistics
	load *config.ServerLoadSpec

	// for all accelerators
	allAllocations map[string]*Allocation

	// allocated solution
	allocation *Allocation

	// current allocation
	curAllocation *Allocation

	spec *config.ServerSpec
}

func NewServerFromSpec(spec *config.ServerSpec) *Server {
	ld := spec.CurrentAlloc.Load
	svcName := spec.Class
	if svcName == "" {
		svcName = config.DefaultServiceClassName
	}
	return &Server{
		name:             spec.Name,
		serviceClassName: svcName,
		modelName:        spec.Model,
		load:             &ld,
		allAllocations:   map[string]*Allocation{},
		curAllocation:    AllocationFromData(&spec.CurrentAlloc),
		spec:             spec,
	}
}

// Calculate allocations for all accelerators
func (s *Server) Calculate(accelerators map[string]*Accelerator) {
	s.allAllocations = make(map[string]*Allocation)
	for _, g := range accelerators {
		if alloc := CreateAllocation(s.name, g.Name()); alloc != nil {
			if s.curAllocation != nil {
				penalty := s.curAllocation.TransitionPenalty(alloc)
				alloc.SetValue(penalty)
			}
			s.allAllocations[g.Name()] = alloc
		}
	}
}

func (s *Server) Name() string {
	return s.name
}

func (s *Server) ServiceClassName() string {
	return s.serviceClassName
}

func (s *Server) Priority() int {
	if svc := GetServiceClass(s.serviceClassName); svc != nil {
		return svc.Priority()
	}
	return config.DefaultServiceClassPriority
}

func (s *Server) ModelName() string {
	return s.modelName
}

func (s *Server) Load() *config.ServerLoadSpec {
	return s.load
}

func (s *Server) SetLoad(load *config.ServerLoadSpec) {
	s.load = load
}

func (s *Server) Allocation() *Allocation {
	return s.allocation
}

func (s *Server) SetAllocation(alloc *Allocation) {
	s.allocation = alloc
	s.UpdateDesiredAlloc()
}

func (s *Server) RemoveAllocation() {
	s.allocation = nil
}

func (s *Server) CurAllocation() *Allocation {
	return s.curAllocation
}

func (s *Server) SetCurAllocation(curAllocation *Allocation) {
	s.curAllocation = curAllocation
}

func (s *Server) AllAllocations() map[string]*Allocation {
	return s.allAllocations
}

func (s *Server) Spec() *config.ServerSpec {
	return s.spec
}

func (s *Server) UpdateDesiredAlloc() {
	if s.allocation != nil {
		s.spec.DesiredAlloc = *s.allocation.AllocationData()
		s.spec.DesiredAlloc.Load = *s.load
	} else {
		s.spec.DesiredAlloc = config.AllocationData{}
	}
}

func (s *Server) ApplyDesiredAlloc() {
	s.spec.CurrentAlloc = s.spec.DesiredAlloc
	s.curAllocation = AllocationFromData(&s.spec.CurrentAlloc)
	s.load = &s.spec.CurrentAlloc.Load
}

func (s *Server) String() string {
	return fmt.Sprintf("Server: name=%s; class=%s; model=%s; load=%v; allocation=%v",
		s.name, s.serviceClassName, s.modelName, s.load, s.allocation)
}
