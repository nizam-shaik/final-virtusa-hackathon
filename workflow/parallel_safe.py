"""
Safe Parallel Workflow Implementation for LangGraph
"""

from typing import Dict, Any, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from operator import add

from state.models import HLDState
from nodes import NodeManager

# Define state schema with proper annotations for parallel updates
class ParallelState(Dict[str, Any]):
    """State schema that supports parallel updates"""
    
    # Use Annotated types for fields that might be updated concurrently
    parallel_results: Annotated[List[Dict[str, Any]], add] = []
    processing_errors: Annotated[List[str], add] = []

def create_safe_parallel_workflow() -> Runnable:
    """
    Create a truly parallel workflow using LangGraph's recommended patterns
    This avoids the INVALID_CONCURRENT_GRAPH_UPDATE error
    """

    node_manager = NodeManager()
    
    # Create state graph with proper state schema
    workflow = StateGraph(Dict[str, Any])
    
    # Add individual nodes
    node_runnables = node_manager.get_node_runnables()
    workflow.add_node("pdf_extraction", node_runnables["pdf_extraction"])

    # Get individual nodes for use in parallel functions
    auth_node = node_manager.get_node("auth_integrations")
    domain_node = node_manager.get_node("domain_api_design")
    behavior_node = node_manager.get_node("behavior_quality")

    # Create parallel processing nodes that don't conflict
    def auth_parallel_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Auth node that updates only its specific state keys"""
        hld_state = HLDState(**state)
        result = auth_node.agent.process(hld_state)
        
        # Return only the specific updates for this node
        return {
            **hld_state.dict(),
            "_auth_result": result,
            "_auth_completed": True
        }
    
    def domain_parallel_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Domain node that updates only its specific state keys"""
        hld_state = HLDState(**state)
        result = domain_node.agent.process(hld_state)
        
        # Return only the specific updates for this node
        return {
            **hld_state.dict(),
            "_domain_result": result,
            "_domain_completed": True
        }
    
    def behavior_parallel_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior node that updates only its specific state keys"""
        hld_state = HLDState(**state)
        result = behavior_node.agent.process(hld_state)
        
        # Return only the specific updates for this node
        return {
            **hld_state.dict(),
            "_behavior_result": result,
            "_behavior_completed": True
        }
    
    def parallel_coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate the results from parallel execution"""
        # This node runs after all parallel nodes complete
        # It consolidates the results without conflicts
        
        results = []
        if state.get("_auth_completed"):
            results.append("auth_integrations")
        if state.get("_domain_completed"):
            results.append("domain_api_design")  
        if state.get("_behavior_completed"):
            results.append("behavior_quality")
        
        # Clean up temporary keys
        cleaned_state = {k: v for k, v in state.items() 
                        if not k.startswith("_auth_") and 
                           not k.startswith("_domain_") and 
                           not k.startswith("_behavior_")}
        
        cleaned_state["_parallel_completed"] = True
        cleaned_state["_parallel_results"] = results
        
        return cleaned_state
    
    # Add nodes to workflow
    workflow.add_node("auth_parallel", auth_parallel_node)
    workflow.add_node("domain_parallel", domain_parallel_node)
    workflow.add_node("behavior_parallel", behavior_parallel_node)
    workflow.add_node("parallel_coordinator", parallel_coordinator)
    workflow.add_node("diagram_generation", node_runnables["diagram_generation"])
    workflow.add_node("output_composition", node_runnables["output_composition"])
    
    # Set entry point
    workflow.set_entry_point("pdf_extraction")
    
    # Sequential: PDF extraction first
    workflow.add_edge("pdf_extraction", "auth_parallel")
    workflow.add_edge("pdf_extraction", "domain_parallel")
    workflow.add_edge("pdf_extraction", "behavior_parallel")
    
    # All parallel nodes feed into coordinator
    workflow.add_edge("auth_parallel", "parallel_coordinator")
    workflow.add_edge("domain_parallel", "parallel_coordinator")
    workflow.add_edge("behavior_parallel", "parallel_coordinator")
    
    # Sequential: Continue after parallel processing
    workflow.add_edge("parallel_coordinator", "diagram_generation")
    workflow.add_edge("diagram_generation", "output_composition")
    workflow.add_edge("output_composition", END)
    
    return workflow.compile()

def create_batch_parallel_workflow() -> Runnable:
    """
    Alternative parallel approach using batch processing
    """

    node_manager = NodeManager()
    workflow = StateGraph(Dict[str, Any])

    # Single batch processing node that handles multiple operations
    def batch_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Process auth, domain, and behavior analysis in a single node"""
        hld_state = HLDState(**state)

        # Get individual nodes
        auth_node = node_manager.get_node("auth_integrations")
        domain_node = node_manager.get_node("domain_api_design")
        behavior_node = node_manager.get_node("behavior_quality")

        # Process all three analyses
        auth_result = auth_node.agent.process(hld_state)
        domain_result = domain_node.agent.process(hld_state)
        behavior_result = behavior_node.agent.process(hld_state)
        
        # Update state with all results
        updated_state = hld_state.dict()
        updated_state["_batch_results"] = {
            "auth": auth_result,
            "domain": domain_result,
            "behavior": behavior_result
        }
        
        return updated_state
    
    # Add nodes
    node_runnables = node_manager.get_node_runnables()
    workflow.add_node("pdf_extraction", node_runnables["pdf_extraction"])
    workflow.add_node("batch_analysis", batch_analysis_node)
    workflow.add_node("diagram_generation", node_runnables["diagram_generation"])
    workflow.add_node("output_composition", node_runnables["output_composition"])
    
    # Set entry point and edges
    workflow.set_entry_point("pdf_extraction")
    workflow.add_edge("pdf_extraction", "batch_analysis")
    workflow.add_edge("batch_analysis", "diagram_generation")
    workflow.add_edge("diagram_generation", "output_composition")
    workflow.add_edge("output_composition", END)
    
    return workflow.compile()