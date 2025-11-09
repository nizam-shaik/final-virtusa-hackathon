from __future__ import annotations
"""
Diagram Generation Agent
"""

# TODO: Implement DiagramAgent class extending BaseAgent
# TODO: Extract diagram plans from domain and behavior states
# TODO: Convert diagram plans to Mermaid syntax using diagram_converter
# TODO: Handle class diagrams (entities, relationships, cardinalities)
# TODO: Handle sequence diagrams (actors, messages, interactions)
# TODO: Set up output directory structure for diagram artifacts
# TODO: Render Mermaid diagrams to images (SVG/PNG) using kroki or mmdc
# TODO: Handle rendering failures gracefully with warnings
# TODO: Collect rendered image paths for HLD embedding
# TODO: Normalize diagram data into DiagramData object
# TODO: Store both Mermaid source code and rendered images
# TODO: Create Diagrams.html interactive viewer
# TODO: Handle configuration options: render_images, image_format, renderer, theme
# TODO: Validate Mermaid syntax before rendering
# TODO: Add error handling for diagram conversion and rendering failures
"""
Diagram Generation Agent
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from state.models import HLDState, DiagramData, ProcessingStatus
from utils.diagram_converter import diagram_plan_to_text
from utils.diagram_renderer import render_diagrams
from diagram_publisher import publish_diagrams

logger = logging.getLogger(__name__)


class DiagramAgent(BaseAgent):
    """
    Generate Mermaid diagrams (class + sequence) from domain and behavior plans,
    render images (optional), and store diagram artifacts into the state's output.
    """

    @property
    def system_prompt(self) -> str:
        # DiagramAgent uses local conversion utilities rather than asking the LLM.
        return "Diagram generation agent: convert diagram plans to Mermaid and render images."

    def process(self, state: HLDState) -> Dict[str, Any]:
        """
        Main entrypoint for diagram generation.
        - Reads diagram plans from state.domain.diagram_plan and state.behavior.diagram_plan
        - Converts to Mermaid (class + sequences)
        - Writes sources and optionally renders images using configured renderer
        - Writes an HTML viewer (full_diagrams.html) via publish_diagrams (preview disabled)
        - Updates state.diagrams and returns a result dict
        """
        state.set_status("diagram_generation", "processing", "Generating diagrams")
        
        logger.info("Starting diagram generation process")

        try:
            # Prepare output directory structure using Project/output path
            project_dir = Path(__file__).resolve().parent.parent
            out_base = Path(state.output.output_dir) if state.output and state.output.output_dir else project_dir / "output" / (state.requirement_name or "unknown")
            diagrams_dir = out_base / "diagrams"
            diagrams_dir.mkdir(parents=True, exist_ok=True)

            # Read config (fallback defaults)
            cfg = state.config or {}
            render_images = bool(cfg.get("render_images", True))
            image_fmt = str(cfg.get("image_format", "svg")).lower()
            renderer = str(cfg.get("renderer", "kroki")).lower()
            theme = str(cfg.get("theme", "default")).lower()
            
            logger.info(f"Configuration: render_images={render_images}, format={image_fmt}, renderer={renderer}, theme={theme}")

            # Compose diagram_plan by merging domain + behavior plans sensibly
            domain_plan = (state.domain.diagram_plan if state.domain and getattr(state.domain, "diagram_plan", None) else {}) or {}
            behavior_plan = (state.behavior.diagram_plan if state.behavior and getattr(state.behavior, "diagram_plan", None) else {}) or {}

            # Merge into single plan expected by diagram_converter
            merged_plan: Dict[str, Any] = {}
            # class plan priority: domain_plan.class or domain_plan.get('class')
            if domain_plan:
                merged_plan.update(domain_plan if isinstance(domain_plan, dict) else {})
            # embed behavior sequences under "sequences"
            if behavior_plan:
                merged_plan.setdefault("sequences", [])
                # behavior_plan may already be a dict with 'sequences'
                if isinstance(behavior_plan, dict) and behavior_plan.get("sequences"):
                    merged_plan["sequences"].extend(behavior_plan.get("sequences"))
                else:
                    # if it's a plain list or something, try to append
                    if isinstance(behavior_plan, list):
                        merged_plan["sequences"].extend(behavior_plan)

            # If merged is empty, attempt to use state.domain and state.behavior directly
            if not merged_plan:
                # best-effort build from domain/behavior
                if state.domain and getattr(state.domain, "diagram_plan", None):
                    merged_plan = state.domain.diagram_plan or {}
                if state.behavior and getattr(state.behavior, "diagram_plan", None):
                    merged_plan.setdefault("sequences", [])
                    merged_plan["sequences"].extend(state.behavior.diagram_plan.get("sequences", []) if isinstance(state.behavior.diagram_plan, dict) else [])

            # Convert plan to mermaid text blocks
            converter_result = diagram_plan_to_text(merged_plan or {})
            if "error" in converter_result:
                raise RuntimeError(f"Diagram conversion error: {converter_result.get('error')}")

            class_text = converter_result.get("class_text", "") or ""
            sequence_texts = converter_result.get("sequence_texts", []) or []

            # Build mermaid_map: a mapping name -> mermaid code
            mermaid_map: Dict[str, str] = {}
            if class_text:
                mermaid_map["class_diagram"] = class_text
            for idx, seq in enumerate(sequence_texts):
                mermaid_map[f"sequence_{idx+1}"] = seq
            
            logger.info(f"Generated {len(mermaid_map)} diagrams: {list(mermaid_map.keys())}")

            # Save mermaid sources and optionally render images
            renderer_results: Dict[str, Dict[str, str]] = {"mmd": {}, "images": {}}
            try:
                if render_images:
                    logger.info(f"Rendering diagrams to {image_fmt} using {renderer}...")
                else:
                    logger.info("Skipping image rendering (render_images=False)")
                    
                # render_diagrams will write sources to diagrams_dir and images into diagrams_dir/img/
                renderer_results = render_diagrams(
                    mermaid_map=mermaid_map,
                    out_dir=str(diagrams_dir),
                    want_images=render_images,
                    image_fmt=image_fmt,
                    renderer=renderer,
                    save_sources=True,
                )
                
                logger.info(f"Saved {len(renderer_results.get('mmd', {}))} Mermaid source files")
                if render_images:
                    logger.info(f"Rendered {len(renderer_results.get('images', {}))} diagram images")
            except Exception as e:
                # Rendering failure should not kill pipeline — record warning and continue
                logger.exception("Diagram rendering failed; continuing with Mermaid sources only.")
                state.add_warning(f"Diagram rendering failed: {e}")

            # Publish full_diagrams.html and optional HLD.html embedding (no Streamlit preview)
            try:
                hld_md = None
                hld_html_out_path = None
                if state.output and getattr(state.output, "hld_html_path", None):
                    hld_html_out_path = state.output.hld_html_path
                publish_results = publish_diagrams(
                    mermaid_map=mermaid_map,
                    out_dir=str(diagrams_dir),
                    title=f"HLD Diagrams - {state.requirement_name or 'HLD'}",
                    theme=theme,
                    preview=False,
                    save_fullpage_html=True,
                    hld_markdown=hld_md,
                    hld_html_out_path=hld_html_out_path,
                )
            except Exception as e:
                logger.exception("Failed to publish diagrams HTML viewer.")
                state.add_warning(f"Failed to publish diagrams viewer: {e}")
                publish_results = {"full_html": None, "hld_html": None}

            # Normalize DiagramData: find image paths (if any) and mermaid texts
            class_image = None
            seq_images: List[str] = []
            images_map = renderer_results.get("images") or {}
            # images_map keys are the same as mermaid_map keys
            if images_map:
                class_image = images_map.get("class_diagram")
                for idx in range(len(sequence_texts)):
                    seq_images.append(images_map.get(f"sequence_{idx+1}"))

            # If render_diagrams didn't return images, attempt to locate files in diagrams_dir/img/
            if not images_map and render_images:
                img_dir = diagrams_dir / "img"
                if img_dir.exists():
                    # try to match names
                    if (img_dir / "class_diagram.svg").exists():
                        class_image = str((img_dir / "class_diagram.svg").resolve())
                    seq_images = []
                    for idx in range(len(sequence_texts)):
                        candidate = img_dir / f"sequence_{idx+1}.{image_fmt}"
                        seq_images.append(str(candidate.resolve()) if candidate.exists() else None)

            # Compose DiagramData
            diagram_data = DiagramData(
                class_text=class_text,
                sequence_texts=sequence_texts,
                class_img_path=class_image,
                seq_img_paths=[s for s in (seq_images or []) if s]
            )

            # Attach to state
            state.diagrams = diagram_data

            # Ensure output paths recorded in state.output
            if not state.output:
                from state.models import OutputData
                state.output = OutputData(output_dir=str(out_base), hld_md_path=state.output.hld_md_path if state.output else "", hld_html_path=state.output.hld_html_path if state.output else "", diagrams_html_path=publish_results.get("full_html"), risk_heatmap_path=state.output.risk_heatmap_path if state.output else None)

            else:
                # update diagrams_html_path
                try:
                    state.output.diagrams_html_path = publish_results.get("full_html")
                except Exception:
                    pass

            state.set_status("diagram_generation", "completed", "Diagrams generated")
            logger.info("[DiagramAgent] Diagram generation finished successfully.")
            return {
                "diagram": diagram_data.dict(),
                "renderer_results": renderer_results,
                "publish_results": publish_results,
            }

        except Exception as exc:
            logger.exception("DiagramAgent failed")
            state.add_error(str(exc))
            state.set_status("diagram_generation", "failed", str(exc))
            return {"error": str(exc)}
