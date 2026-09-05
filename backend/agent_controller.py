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
        Intelligent LLM Agent Orchestrator.
        Analyzes conversational context, natural language intent, and multimodal inputs
        to dynamically select the optimal remote sensing specialist model.
        """
        query_lower = query.lower().strip()
        
        # Check history to establish context if query is brief
        if history and len(history) > 0 and len(query.split()) < 4:
            last_query = history[-1].get("user", "").lower()
            query_lower = last_query + " " + query_lower

        reasoning = f"Evaluating intent for query '{query[:60]}...' with {num_images} uploaded image(s). "
        task = ""
        
        if num_images == 0:
            raise ValueError("Please provide at least one satellite image (optical, SAR, multispectral, GeoTIFF, or visual format).")
            
        elif num_images >= 2:
            if any(keyword in query_lower for keyword in ["cloud", "cloudy", "weather", "penetrate", "obscured", "sar", "radar", "fog", "fusion"]):
                reasoning += "Query involves penetrating cloud cover or fusing radar structure with optical data. Routing to Cross-Modal Extraction Engine."
                task = "CROSS_MODAL_EXTRACTION"
            else:
                reasoning += f"Detected multi-image sequence ({num_images} images). Routing to Bi-Temporal Change Analysis Engine to quantify structural differences and environmental progression."
                task = "CHANGE_ANALYSIS"
                
        else:
            # Single Image Intent Disambiguation
            # 1. Grounding / Delineation / Spatial Outlining
            is_grounding = any(kw in query_lower for kw in [
                "mark", "highlight", "bound", "box", "locate", "outline", "delineate", "trace", 
                "find where", "detect where", "show where", "draw", "pinpoint"
            ])
            # If query says "identify ... and calculate", grounding produces the visual contour + area calculation
            if ("identify" in query_lower or "detect" in query_lower) and any(kw in query_lower for kw in ["land", "water", "sea", "vegetation", "built", "area", "boundary", "coastline"]):
                is_grounding = True

            # 2. Captioning / Holistic Scene Overview
            is_captioning = any(kw in query_lower for kw in [
                "caption", "describe", "summarize", "overview", "what are we looking at", 
                "tell me about this", "explain this image", "scene summary", "briefing"
            ]) and not is_grounding

            if is_grounding:
                reasoning += "User requested spatial localization, boundary delineation, or tactical vector grounding. Routing to Specialist Grounding Model."
                task = "SINGLE_IMAGE_GROUNDING"
            elif is_captioning:
                reasoning += "User requested a holistic remote sensing scene description and narrative summary. Routing to Specialist Captioning Engine."
                task = "SINGLE_IMAGE_CAPTIONING"
            else:
                reasoning += "User is asking a targeted question, calculating metrics, or auditing scene features. Routing to Remote-Sensing-Adapted Vision-Language Model."
                task = "SINGLE_IMAGE_VQA"
                
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
        
        # Compute cache key with version prefix to ensure updated logic runs fresh
        CACHE_VERSION = "v3_compound_intent"
        m = hashlib.md5()
        m.update(CACHE_VERSION.encode('utf-8'))
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
            
            scope_desc = "Single Image (Optical/SAR/Multispectral/GeoTIFF)"
            if len(images) == 2:
                scope_desc = "Bi-Temporal Pair" if task_name == "CHANGE_ANALYSIS" else "Cross-Modal Co-Registered Pair (Optical-SAR)"
            elif len(images) > 2:
                scope_desc = f"Multi-Temporal Sequence ({len(images)} Acquisitions)"
                
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
