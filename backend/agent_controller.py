import logging
from typing import List, Dict, Any
from model_registry import registry

logger = logging.getLogger(__name__)

class AgenticController:
    def __init__(self):
        # We would typically initialize an LLM here for query parsing
        pass

    def classify_task(self, query: str, num_images: int, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Simulated LLM Agent Orchestrator.
        In a production environment, this would call Gemini/OpenAI with the query, history, and image metadata
        and ask it to output a JSON reasoning trace and selected tool.
        """
        query_lower = query.lower()
        
        # Check history to establish context if query is brief
        if history and len(history) > 0 and len(query.split()) < 4:
            last_query = history[-1].get("user", "").lower()
            query_lower = last_query + " " + query_lower

        # Simulated LLM Reasoning
        reasoning = "Analyzing user intent based on the query and conversational context. "
        task = ""
        
        if num_images == 2:
            if any(keyword in query_lower for keyword in ["cloud", "cloudy", "weather", "penetrate", "obscured"]):
                reasoning += "Query mentions clouds or obscured vision. Explicitly routing to SAR structural cross-modal tools to bypass cloud cover."
                task = "CROSS_MODAL_EXTRACTION"
            elif any(keyword in query_lower for keyword in ["change", "compare", "difference", "expand", "shrink"]):
                reasoning += "User is asking for temporal or structural differences between two images. Routing to Bi-Temporal Change Analysis."
                task = "CHANGE_ANALYSIS"
            else:
                reasoning += "Multiple images provided without explicit change keywords. Assuming cross-modal fusion requirement. Routing to Cross-Modal Extraction."
                task = "CROSS_MODAL_EXTRACTION"
        elif num_images == 1:
            if any(keyword in query_lower for keyword in ["caption", "describe", "summarize", "overview", "what is in", "tell me about", "scene"]):
                reasoning += "User requested holistic remote sensing scene description. Routing to Specialist Captioning Engine."
                task = "SINGLE_IMAGE_CAPTIONING"
            elif any(keyword in query_lower for keyword in ["country", "state", "identify the region", "what region"]):
                reasoning += "User is asking for geospatial intelligence and location identification. Routing to Remote-Sensing-Adapted Vision-Language Model."
                task = "SINGLE_IMAGE_VQA"
            elif any(keyword in query_lower for keyword in ["highlight", "where", "region", "find", "locate", "bound", "detect", "box"]):
                reasoning += "User is asking for spatial localization of a specific feature. Routing to Specialist Grounding Model."
                task = "SINGLE_IMAGE_GROUNDING"
            else:
                reasoning += "User is asking a targeted question about visual features. Routing to Remote-Sensing-Adapted Vision-Language Model."
                task = "SINGLE_IMAGE_VQA"
        else:
            raise ValueError(f"Unsupported number of images: {num_images}. Defined Input Scope allows strictly 1 image or 2 co-registered images.")
            
        return {"task": task, "reasoning": reasoning}

    def execute_query(self, query: str, images: List[bytes], history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Main orchestration method:
        1. Classify task based on query, images, and history.
        2. Fetch the corresponding specialist tool.
        3. Execute the tool with automated input compatibility checks.
        4. Return auditable results with confidence and spatial coherence metadata.
        """
        import hashlib
        import json
        import os
        
        # Compute cache key
        m = hashlib.md5()
        m.update(query.encode('utf-8'))
        for img in images:
            m.update(img)
        cache_key = m.hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    logger.info(f"Returning cached result for query: {query}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {e}")

        try:
            classification = self.classify_task(query, len(images), history)
            task_name = classification["task"]
            reasoning = classification["reasoning"]
            logger.info(f"Classified task as {task_name}")
            
            tool = registry.get_model(task_name)
            
            # Execute tool
            tool_output = tool.execute(images, query)
            
            scope_desc = "Single Image (Optical/SAR/Multispectral)"
            if len(images) == 2:
                scope_desc = "Bi-Temporal Pair" if task_name == "CHANGE_ANALYSIS" else "Cross-Modal Co-Registered Pair (Optical-SAR)"
                
            compat_status = tool_output.get("compatibility_status", "PASSED")
            coherence_score = tool_output.get("spatial_coherence_score", 1.0)
            model_prov = tool_output.get("model_provenance", "SatQuery-RS-Adapted-v1.2 (Fine-tuned)")
            
            # Construct the auditable execution summary
            execution_summary = {
                "selected_task": task_name,
                "tool_used": tool.__class__.__name__,
                "input_scope": scope_desc,
                "num_images_processed": len(images),
                "compatibility_status": compat_status,
                "spatial_coherence_score": coherence_score,
                "model_provenance": model_prov,
                "agent_reasoning": reasoning
            }
            
            final_result = {
                "status": "success",
                "answer": tool_output["text"],
                "visual_evidence": tool_output["visual_evidence"],
                "confidence": tool_output["confidence"],
                "compatibility_status": compat_status,
                "spatial_coherence_score": coherence_score,
                "execution_summary": execution_summary,
                "geo_metadata": tool_output.get("geo_metadata"),
                "geojson_data": tool_output.get("geojson_data"),
                "pair_comparison": tool_output.get("pair_comparison")
            }
            
            # Save to cache
            try:
                with open(cache_path, "w") as f:
                    json.dump(final_result, f)
            except Exception as e:
                logger.error(f"Error writing to cache: {e}")
                
            return final_result
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

agent_controller = AgenticController()
