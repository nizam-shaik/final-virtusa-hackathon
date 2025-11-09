from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import workflow factory (uses HLDWorkflow implemented earlier)
from workflow.hld_workflow import create_hld_workflow

DATA_DIR = Path(__file__).resolve().parent / "data"  # Project/data


# -----------------------
# Helpers
# -----------------------
def list_requirement_pdfs(folder: Path = DATA_DIR) -> List[Path]:
    """List PDF files in the data folder sorted by name."""
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])


def get_pdf_info(pdf_path: Path) -> Dict[str, Any]:
    """Return simple metadata about the PDF for UI display."""
    try:
        st_size = pdf_path.stat().st_size
        return {
            "name": pdf_path.name,
            "path": str(pdf_path),
            "size_kb": round(st_size / 1024, 1),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pdf_path.stat().st_mtime)),
        }
    except Exception:
        return {"name": pdf_path.name, "path": str(pdf_path), "size_kb": None, "modified": None}


def safe_load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------
# UI renderers (concise)
# -----------------------
def render_markdown_section(title: str, md_text: str):
    st.subheader(title)
    if md_text:
        st.markdown(md_text)
    else:
        st.info("No content available.")

            
def render_key_value(title: str, data: Dict):
    st.subheader(title)
    if not data:
        st.info("No data.")
        return
    for k, v in data.items():
        st.write(f"**{k}**: {v}")


def render_list_section(title: str, items: List[str]):
    st.subheader(title)
    if not items:
        st.info("No items.")
        return
    for item in items:
        st.markdown(f"- {item}")


def render_json_pretty(title: str, obj: Any):
    st.subheader(title)
    if not obj:
        st.info("No data.")
        return
    st.json(obj)


# -----------------------
# Workflow runner (streaming)
# -----------------------
def run_workflow_and_stream(pdf_path: str, workflow_type: str, config: Dict[str, Any]):
    """
    Run workflow.stream() and yield node-by-node events.
    This function must be executed synchronously from Streamlit (we use asyncio.run).
    """
    logger.info("=" * 80)
    logger.info("WORKFLOW EXECUTION STARTED")
    logger.info("=" * 80)
    logger.info(f"PDF Path: {pdf_path}")
    logger.info(f"Workflow Type: {workflow_type}")
    logger.info(f"Configuration: {config}")
    logger.info("=" * 80)
    
    wf = create_hld_workflow(workflow_type)

    # Input object can be a dict
    input_data = {"pdf_path": pdf_path, "config": config}

    # Use asyncio to consume the async generator
    async def _consume():
        events = []
        async for event in wf.stream(input_data):
            events.append(event)
        return events

    # Be friendly with environments where an event loop might already exist
    try:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            events = loop.run_until_complete(_consume())
        finally:
            try:
                loop.close()
            except Exception:
                pass
        return events
    except Exception as e:
        try:
            # Fallback to asyncio.run if fresh loop strategy fails
            events = asyncio.run(_consume())
            return events
        except Exception as e2:
            return [{"type": "error", "message": f"{e2}"}]


# -----------------------
# Main Streamlit App
# -----------------------
def main():
    st.set_page_config(page_title="DesignMind GenAI — HLD Generator", layout="wide")
    st.title("DesignMind GenAI — HLD Generator")

    # Sidebar: configuration
    st.sidebar.header("Configuration")
    workflow_type = st.sidebar.selectbox("Workflow type", ["parallel", "sequential", "conditional"], index=0)
    render_images = st.sidebar.checkbox("Render diagrams to images", value=False)
    image_format = st.sidebar.selectbox("Image format", ["svg", "png"], index=0)
    renderer = st.sidebar.selectbox("Renderer", ["kroki", "mmdc"], index=1)
    theme = st.sidebar.selectbox("Diagram theme", ["default", "neutral", "dark"], index=0)

    config = {
        "render_images": render_images,
        "image_format": image_format,
        "renderer": renderer,
        "theme": theme,
    }
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("CONFIGURATION SETTINGS")
    logger.info("=" * 80)
    logger.info(f"Workflow Type: {workflow_type}")
    logger.info(f"Render Images: {render_images}")
    logger.info(f"Image Format: {image_format}")
    logger.info(f"Renderer: {renderer}")
    logger.info(f"Diagram Theme: {theme}")
    logger.info("=" * 80)

    tabs = st.tabs(["HLD Generation", "ML Training", "Quality Prediction"])

    # -------------------------
    # Tab 1: HLD Generation
    # -------------------------
    with tabs[0]:
        st.header("HLD Generation")
        pdfs = list_requirement_pdfs()
        if not pdfs:
            st.warning("No requirement PDFs found in data/ — place PDFs in the data/ folder.")
            return

        pdf_names = [p.name for p in pdfs]
        default_idx = 0
        selection = st.selectbox("Select requirement PDF", pdf_names, index=default_idx)
        selected_pdf = pdfs[pdf_names.index(selection)]
        info = get_pdf_info(selected_pdf)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Selected:** {info['name']}")
            st.write(f"Path: `{info['path']}`")
            st.write(f"Size: {info['size_kb']} KB")
            st.write(f"Modified: {info['modified']}")
        with col2:
            st.download_button("Download PDF", data=selected_pdf.read_bytes(), file_name=info["name"])

        # Check if output already exists for this requirement
        output_dir = Path(__file__).resolve().parent / "output" / Path(selected_pdf).stem
        output_exists = (output_dir / "json").exists() and (output_dir / "json" / "authentication.json").exists()
        
        if output_exists:
            st.success(f"✅ Output already exists for {Path(selected_pdf).stem}")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                view_existing = st.button("📂 View Existing Output", type="primary", use_container_width=True)
            with col_b:
                regenerate = st.button("🔄 Regenerate HLD", type="secondary", use_container_width=True)
        else:
            st.info(f"ℹ️ No existing output found for {Path(selected_pdf).stem}")
            regenerate = st.button("Generate HLD", type="primary", use_container_width=True)
            view_existing = False

        # placeholders for progress and outputs
        progress_box = st.empty()
        log_box = st.empty()
        outputs_box = st.container()
        
        # Determine if we should show output
        show_output = view_existing or (output_exists and not regenerate)

        if regenerate:
            progress_box.info("Starting workflow... (this may take a while)")
            # Run streaming workflow; update UI per event
            events = run_workflow_and_stream(str(selected_pdf), workflow_type, config)

            # Render events log
            for e in events:
                etype = e.get("type")
                if etype == "node_start":
                    log_box.info(f"Started: {e.get('node')}")
                elif etype == "node_complete":
                    log_box.success(f"Completed: {e.get('node')} ({e.get('status')}) in {e.get('duration')}s")
                elif etype == "error":
                    log_box.error(f"Error: {e.get('message')}")
                elif etype == "started":
                    log_box.info("Workflow started.")
                elif etype == "completed":
                    log_box.success("Workflow completed.")
                else:
                    log_box.write(f"{e}")

            # After complete, try to read output paths from the last state (best-effort)
            # We expect the last event to include final state if using NodeManager stream.
            final_state = None
            for e in reversed(events):
                if isinstance(e.get("state"), dict):
                    final_state = e.get("state")
                    break
            
            # Mark that we should show output after generation
            show_output = True
        else:
            # Not regenerating, just viewing existing
            final_state = None
        
        # Show output if either viewing existing or just generated
        if show_output or output_exists:
            with outputs_box:
                st.markdown("## Results")
                
                # Determine output directory for the selected requirement once
                output_dir = Path(__file__).resolve().parent / "output" / Path(selected_pdf).stem

                st.markdown("## 📋 Extracted Requirements")

                # 1) Extracted requirements (try final_state -> json file)
                extracted_md = None
                try:
                    extracted_obj = final_state.get("extracted") if final_state else None
                    if extracted_obj:
                        if isinstance(extracted_obj, str):
                            extracted_md = extracted_obj
                        elif isinstance(extracted_obj, dict):
                            extracted_md = extracted_obj.get("markdown") or extracted_obj.get("markdown_text") or extracted_obj.get("text")
                except Exception:
                    extracted_md = None

                if not extracted_md:
                    extracted_path = output_dir / "json" / "extracted.json"
                    if extracted_path.exists():
                        try:
                            ex = safe_load_json(str(extracted_path))
                            if isinstance(ex, dict):
                                extracted_md = ex.get("markdown") or ex.get("markdown_text") or json.dumps(ex, indent=2)
                            else:
                                extracted_md = str(ex)
                        except Exception:
                            extracted_md = None

                if extracted_md:
                    with st.expander("📄 View Extracted Content (Click to Expand/Minimize)", expanded=False):
                        # Add scrolling container with max height
                        st.markdown(
                            f'<div style="max-height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">{extracted_md}</div>',
                            unsafe_allow_html=True
                        )
                # Get authentication data from final_state or load from JSON file
                auth = None
                if final_state:
                    auth = final_state.get("authentication") or {}
                
                # Fallback: Load from authentication.json if not in state
                if not auth or not any(auth.values()):
                    auth_json_path = output_dir / "json" / "authentication.json"
                    if auth_json_path.exists():
                        try:
                            auth = safe_load_json(str(auth_json_path)) or {}
                        except Exception as e:
                            st.warning(f"Could not load authentication.json: {e}")
                            auth = {}
                
                st.markdown("## 🔐 Authentication & Security")
                if not auth or not any(auth.values()):
                    st.info("No authentication information available.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Authentication Methods")
                        # Tolerant fallbacks: auth_methods OR flows OR methods
                        methods = auth.get("auth_methods") or auth.get("flows") or auth.get("methods") or []
                        if methods:
                            for method in methods:
                                st.write(f"✓ {method}")
                        else:
                            st.caption("No authentication methods specified")
                        
                        # OAuth providers OR idp_options
                        st.markdown("#### OAuth / IDP Providers")
                        providers = auth.get("oauth_providers") or auth.get("idp_options") or []
                        if providers:
                            for provider in providers:
                                st.write(f"• {provider}")
                        else:
                            st.caption("No OAuth providers specified")
                    
                    with col2:
                        # Security features
                        st.markdown("#### Security Features")
                        sec_features = auth.get("security_features") or auth.get("features") or []
                        if sec_features:
                            for feature in sec_features:
                                st.write(f"🔒 {feature}")
                        else:
                            st.caption("No security features specified")
                        
                        # Compliance requirements
                        st.markdown("#### Compliance Requirements")
                        compliance = auth.get("compliance") or auth.get("compliance_requirements") or []
                        if compliance:
                            for req in compliance:
                                st.write(f"📋 {req}")
                        else:
                            st.caption("No compliance requirements specified")
                    
                    # Additional sections
                    if auth.get("security_policies"):
                        st.markdown("#### Security Policies")
                        for policy in auth.get("security_policies", []):
                            st.write(f"📜 {policy}")
                    
                    if auth.get("additional_notes"):
                        st.markdown("#### Additional Security Notes")
                        st.info(auth.get("additional_notes"))
                    
                    # Fallback: show actors / threats (alternative fields in authentication.json)
                    if auth.get("actors"):
                        st.markdown("#### Actors / Users")
                        actors_cols = st.columns(3)
                        for idx, actor in enumerate(auth.get("actors", [])):
                            with actors_cols[idx % 3]:
                                st.write(f"👤 {actor}")
                    
                    if auth.get("threats"):
                        st.markdown("#### Security Threats")
                        for threat in auth.get("threats", []):
                            st.write(f"⚠️ {threat}")

                # Integrations
                integrations = None
                if final_state:
                    integrations = final_state.get("integrations") or []

                # Fallback: Load from integrations.json if not in state
                if not integrations:
                    integrations_json_path = output_dir / "json" / "integrations.json"
                    if integrations_json_path.exists():
                        try:
                            integrations = safe_load_json(str(integrations_json_path)) or []
                        except Exception:
                            integrations = []

                st.subheader("🔗 Integrations")
                if not integrations:
                    st.info("No integrations available.")
                else:
                    # Create a dataframe for integrations
                    integration_data = []
                    for integration in integrations:
                        if isinstance(integration, dict):
                            # tolerant field mapping / fallbacks
                            name = integration.get("name") or integration.get("system") or integration.get("system_name") or ""
                            typ = integration.get("type") or integration.get("protocol") or ""
                            purpose = integration.get("purpose") or integration.get("description") or ""
                            status = integration.get("status") or integration.get("state") or integration.get("auth") or ""
                            row = {
                                "Integration Name": name,
                                "Type": typ,
                                "Purpose": purpose,
                                "Status": status,
                            }
                            integration_data.append(row)
                    if integration_data:
                        df_integrations = pd.DataFrame(integration_data)
                        st.dataframe(df_integrations, use_container_width=True)

                # Domain
                domain = None
                if final_state:
                    domain = final_state.get("domain") or {}

                # Fallback: Load from domain.json if not in state
                if not domain:
                    domain_json_path = output_dir / "json" / "domain.json"
                    if domain_json_path.exists():
                        try:
                            domain = safe_load_json(str(domain_json_path)) or {}
                        except Exception:
                            domain = {}

                st.subheader("📦 Domain Entities")

                # Check if we have entities or APIs to display
                entities = domain.get("entities", []) if domain else []
                apis = domain.get("apis", []) if domain else []

                if not entities and not apis:
                    st.warning("⚠️ No domain entities or APIs found. The domain.json file may be empty or not generated properly.")
                    # Show what's in domain.json for debugging
                    if domain:
                        with st.expander("🔍 Debug: View domain.json content"):
                            st.json(domain)
                else:
                    # Display Entities
                    if entities:
                        entity_cols = st.columns(2)
                        for idx, entity in enumerate(entities):
                            with entity_cols[idx % 2]:
                                if isinstance(entity, dict):
                                    entity_name = entity.get("name", "Unknown Entity")
                                    entity_attrs = entity.get("attributes", [])
                                else:
                                    entity_name = str(entity)
                                    entity_attrs = []

                                with st.expander(f"📦 {entity_name}"):
                                    if entity_attrs:
                                        st.markdown("**Attributes:**")
                                        # Display each attribute with compact spacing
                                        for attr in entity_attrs:
                                            st.text(f"• {attr}")
                                    else:
                                        st.info("No attributes defined")
                    else:
                        st.info("No entities found in domain data.")

                    # Display APIs
                    if apis:
                        st.markdown("### 🌐 REST APIs")
                        with st.expander(f"View {len(apis)} API Endpoints", expanded=False):
                            for idx, api in enumerate(apis):
                                if isinstance(api, dict):
                                    api_name = api.get("name", "Unnamed API")
                                    # Clean up API name: replace // with - and handle URL formatting
                                    if isinstance(api_name, str):
                                        api_name = api_name.replace("//", " - ")
                                    api_desc = api.get("description", "No description")
                                    api_request = api.get("request", {})
                                    api_response = api.get("response", {})

                                    # API header using native Streamlit components
                                    with st.container():
                                        st.markdown(f"#### 🔌 {api_name}")
                                        st.caption(api_desc)

                                        # Request and Response in columns
                                        api_detail_cols = st.columns(2)

                                        with api_detail_cols[0]:
                                            st.markdown("**📤 Request**")
                                            if api_request:
                                                if isinstance(api_request, dict):
                                                    # Display as formatted JSON
                                                    st.json(dict(list(api_request.items())[:10]))
                                                else:
                                                    st.code(str(api_request), language="json")
                                            else:
                                                st.info("No request body")

                                        with api_detail_cols[1]:
                                            st.markdown("**📥 Response**")
                                            if api_response:
                                                if isinstance(api_response, dict):
                                                    # Display as formatted JSON
                                                    st.json(dict(list(api_response.items())[:10]))
                                                else:
                                                    st.code(str(api_response), language="json")
                                            else:
                                                st.info("No response body")

                                        # Add separator between APIs
                                        if idx < len(apis) - 1:
                                            st.divider()
                # Behavior
                behavior = None
                if final_state:
                    behavior = final_state.get("behavior") or {}

                # Fallback: Load from behavior.json if not in state
                if not behavior or not any(behavior.values()):
                    behavior_json_path = output_dir / "json" / "behavior.json"
                    if behavior_json_path.exists():
                        try:
                            behavior = safe_load_json(str(behavior_json_path)) or {}
                        except Exception:
                            behavior = {}

                st.subheader("🎯 Use Cases")
                if not behavior or not any(behavior.values()):
                    st.info("No behavior information available.")
                else:
                    if "use_cases" in behavior:
                        use_cases = behavior.get("use_cases", [])
                        with st.expander(f"View {len(use_cases)} Use Cases", expanded=False):
                            for uc in use_cases:
                                if isinstance(uc, str):
                                    name = uc
                                    steps = None
                                    sequence_flow = None
                                elif isinstance(uc, dict):
                                    name = uc.get("name") or uc.get("title") or "Unnamed Use Case"
                                    steps = uc.get("steps")
                                    sequence_flow = uc.get("sequence_flow") or uc.get("sequence") or uc.get("sequence_steps")
                                else:
                                    name = str(uc)
                                    steps = None
                                    sequence_flow = None

                                st.write(f"📋 **{name}**")
                                if steps:
                                    st.markdown("**Steps:**")
                                    for step in steps:
                                        st.markdown(f"- {step}")
                                if sequence_flow:
                                    st.markdown("**Sequence Flow:**")
                                    for flow in sequence_flow:
                                        st.markdown(f"1. {flow}")

                    # NFRs in a clean table format
                    st.subheader("⚡ Non-Functional Requirements")
                    if "nfrs" in behavior:
                        nfrs_obj = behavior.get("nfrs")
                        if isinstance(nfrs_obj, dict):
                            # category -> list with expanders
                            for cat, items in nfrs_obj.items():
                                # Add category-specific emojis
                                cat_emoji = {
                                    'performance': '🚀',
                                    'security': '🔒',
                                    'scalability': '📈',
                                    'reliability': '🛡️',
                                    'usability': '👥',
                                    'maintainability': '🔧',
                                    'availability': '⏱️',
                                }.get(cat.lower(), '📋')

                                with st.expander(f"{cat_emoji} {cat.capitalize()}", expanded=False):
                                    if not items:
                                        st.info("No items specified.")
                                    else:
                                        for it in items:
                                            if isinstance(it, dict):
                                                req = it.get("requirement") or it.get("desc") or str(it)
                                                target = it.get("target", "N/A")
                                                st.markdown(
                                                    f"""
                                                    **Requirement:** {req}  
                                                    **Target:** `{target}`
                                                    """
                                                )
                                                st.divider()
                                            else:
                                                st.markdown(f"• {it}")
                        else:
                            # assume list - display in cards
                            with st.expander(f"📋 View {len(nfrs_obj or [])} Requirements", expanded=False):
                                nfr_cols = st.columns(2)
                                for i, nfr in enumerate(nfrs_obj or []):
                                    with nfr_cols[i % 2]:
                                        # Handle both dict and string formats
                                        if isinstance(nfr, dict):
                                            category = nfr.get('category', 'General')
                                            requirement = nfr.get('requirement', 'N/A')
                                            target = nfr.get('target', 'N/A')

                                            st.markdown(
                                                f"""
                                                **{category}**  
                                                🎯 {requirement}  
                                                📊 Target: `{target}`
                                                """
                                            )
                                        else:
                                            # If nfr is a string, display it as is
                                            st.markdown(f"• {str(nfr)}")

                                        if i < len(nfrs_obj) - 1 and i % 2 == 1:
                                            st.markdown("---")

                    # Risks with severity indicators
                    st.subheader("⚠️ Risk Analysis")
                    if "risks" in behavior:
                        risks_list = behavior.get("risks", [])
                        if not risks_list:
                            st.info("No risks identified.")
                        else:
                            with st.expander(f"View {len(risks_list)} Identified Risks", expanded=False):
                                for idx, risk in enumerate(risks_list):
                                    # Handle both dict and string formats
                                    if isinstance(risk, dict):
                                        name = risk.get('name') or risk.get('desc') or 'Unnamed Risk'
                                        impact = risk.get('impact', 'N/A')
                                        mitigation = risk.get('mitigation', 'N/A')
                                        assumption = risk.get('assumption') or risk.get('assumptions')
                                    else:
                                        # If risk is a string, display it as is
                                        name = str(risk)
                                        impact = 'N/A'
                                        mitigation = 'N/A'
                                        assumption = None

                                    # Display risk in a container
                                    with st.container():
                                        st.markdown(f"**{name}**")

                                        st.markdown(
                                            f"""
                                            **💥 Impact:** {impact}  
                                            **🛡️ Mitigation:** {mitigation}
                                            """
                                        )

                                        if assumption:
                                            st.markdown(f"**💭 Assumption:** {assumption}")

                                    # Add separator between risks
                                    if idx < len(risks_list) - 1:
                                        st.divider()

                    # Risk heatmap display in expander
                    risk_img = output_dir / "diagrams" / "risk_heatmap.png"
                    if risk_img.exists():
                        st.markdown("### 📊 Risk Heatmap")
                        with st.expander("View Risk Heatmap", expanded=True):
                            st.image(str(risk_img), width=600)
                            st.download_button(
                                "⬇️ Download Risk Heatmap",
                                data=risk_img.read_bytes(),
                                file_name="risk_heatmap.png",
                                help="Download the risk heatmap visualization",
                            )
                    # Diagrams area: show links and embedded diagrams if available
                out_obj = {}
                if final_state:
                    out_obj = final_state.get("output") or {}

                hld_md_path = None
                hld_html_path = None
                diagrams_html_path = None

                if isinstance(out_obj, dict):
                    hld_md_path = out_obj.get("hld_md_path") or out_obj.get("hld_md")
                    hld_html_path = out_obj.get("hld_html_path") or out_obj.get("hld_html")
                    diagrams_html_path = out_obj.get("diagrams_html_path") or out_obj.get("diagrams_html")

                # If paths not in final_state, try to find them in output directory
                if not hld_md_path:
                    potential_hld_md = output_dir / "hld" / "HLD.md"
                    if potential_hld_md.exists():
                        hld_md_path = str(potential_hld_md)

                if not hld_html_path:
                    potential_hld_html = output_dir / "hld" / "HLD.html"
                    if potential_hld_html.exists():
                        hld_html_path = str(potential_hld_html)

                if not diagrams_html_path:
                    potential_diagrams = output_dir / "diagrams" / "full_diagrams.html"
                    if potential_diagrams.exists():
                        diagrams_html_path = str(potential_diagrams)

                # Normalize paths (resolve relative → absolute) and provide links
                if hld_md_path:
                    try:
                        hld_md_path = str(Path(hld_md_path).resolve())
                    except Exception:
                        pass
                if hld_html_path:
                    try:
                        hld_html_path = str(Path(hld_html_path).resolve())
                    except Exception:
                        pass
                if diagrams_html_path:
                    try:
                        diagrams_html_path = str(Path(diagrams_html_path).resolve())
                    except Exception:
                        pass
                # fallback: check output dir for a full_diagrams.html viewer
                try:
                    if not diagrams_html_path:
                        fallback_viewer = output_dir / "diagrams" / "full_diagrams.html"
                        if fallback_viewer.exists():
                            diagrams_html_path = str(fallback_viewer.resolve())
                except Exception:
                    pass

                # Interactive Diagrams first
                if diagrams_html_path and Path(diagrams_html_path).exists():
                    st.markdown("## 📊 Interactive Diagrams")
                    with st.expander("🔍 View Diagrams (Click to Expand/Minimize)", expanded=True):
                        try:
                            # Read the HTML content and embed it directly with white background
                            html_content = Path(diagrams_html_path).read_text(encoding='utf-8')
                            # Wrap content with white background styling
                            styled_html = f"""
                            <div style="background-color: white; padding: 20px; border-radius: 5px;">
                                {html_content}
                            </div>
                            """
                            st.components.v1.html(styled_html, height=800, scrolling=True)
                        except Exception as e:
                            st.error(f"Could not render diagrams: {e}")
                            st.markdown(f"[Open Diagrams Viewer]({Path(diagrams_html_path).as_uri()})")

                # HLD Document below
                if hld_md_path and Path(hld_md_path).exists():
                    st.markdown("## 📄 HLD Document")
                    with st.expander("📖 View HLD Markdown (Click to Expand/Minimize)", expanded=False):
                        try:
                            hld_content = Path(hld_md_path).read_text()
                            # Add scrolling container with max height
                            st.markdown(
                                f'<div style="max-height: 600px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">{hld_content}</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            st.error("Could not read HLD content")
                    st.download_button(
                        "⬇️ Download HLD.md",
                        data=Path(hld_md_path).read_bytes(),
                        file_name=Path(hld_md_path).name,
                        help="Download the High Level Design document in Markdown format",
                    )


    # -------------------------
    # Tab 2: ML Training (integrated)
    # -------------------------
    with tabs[1]:
        st.header("ML Training (Dataset & Models)")
        st.info("Preview the synthetic dataset and train ML models.")
        from ml.training.train_large_model import LargeScaleMLTrainer
        dataset_path = os.path.join(os.path.dirname(__file__), "ml", "training", "synthetic_hld_dataset.csv")
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            logger.info(f"Loaded dataset from {dataset_path} with shape {df.shape}")
        else:
            st.warning("Synthetic dataset not found. Please generate it first.")
            logger.warning(f"Dataset not found at {dataset_path}")

        trainer = None
        if st.button("Train models", type="primary"):
            logger.info("Training models started.")
            trainer = LargeScaleMLTrainer()
            trainer.load_dataset(dataset_path)
            logger.info("Dataset loaded for training.")
            trainer.prepare_data()
            logger.info("Data prepared (train/test split).")
            trainer.train_models()
            logger.info("Models trained.")
            trainer.evaluate_models()
            logger.info("Model evaluation complete.")
            out_dir = os.path.join(os.path.dirname(__file__), "ml", "training", "models")
            trainer.save_models(out_dir)
            logger.info(f"Models saved to {out_dir}")
            st.success("Models trained and saved!")
            st.subheader("Model Metrics (Test Set)")
            # Present metrics in a neat, readable table
            try:
                df_metrics = pd.DataFrame.from_dict(trainer.results, orient='index')
                df_metrics.index.name = 'Model'
                df_metrics = df_metrics.reset_index()
                # Format columns
                fmt = {
                    'R2': '{:.3f}',
                    'RMSE': '{:.2f}',
                    'MAE': '{:.2f}',
                    'MAPE': '{:.1f}'
                }
                st.dataframe(df_metrics.style.format(fmt), use_container_width=True)
                # Highlight best model by R2
                try:
                    best = df_metrics.sort_values('R2', ascending=False).iloc[0]
                    st.success(f"Best model: {best['Model']} — R2 {best['R2']:.3f}, RMSE {best['RMSE']:.2f}, MAE {best['MAE']:.2f}")
                except Exception:
                    pass
            except Exception:
                # Fallback to key-value display per model
                for name, metrics in trainer.results.items():
                    colA, colB, colC, colD = st.columns(4)
                    with colA:
                        st.metric(f"{name} R2", f"{metrics.get('R2', 0):.3f}")
                    with colB:
                        st.metric(f"{name} RMSE", f"{metrics.get('RMSE', 0):.2f}")
                    with colC:
                        st.metric(f"{name} MAE", f"{metrics.get('MAE', 0):.2f}")
                    with colD:
                        st.metric(f"{name} MAPE", f"{metrics.get('MAPE', 0):.1f}%")
            # Log metrics
            for name, metrics in trainer.results.items():
                logger.info(f"Metrics for {name}: {metrics}")

    # -------------------------
    # Tab 3: Quality Prediction (integrated)
    # -------------------------
    with tabs[2]:
        st.header("Quality Prediction")
        st.info("Load trained models and predict HLD quality scores.")
        from ml.training.inference import HLDQualityPredictor
        from ml.models.feature_extractor import FeatureExtractor
        model_dir = os.path.join(os.path.dirname(__file__), "ml", "training", "models")
        feature_extractor = FeatureExtractor()
        predictor = HLDQualityPredictor(model_dir, feature_extractor.get_feature_names())
        models_loaded = predictor.load_models_from_disk()
        if not models_loaded:
            st.warning("No trained models found. Please train models first.")
            logger.warning(f"No trained models found in {model_dir}")
        else:
            st.success("Models loaded!")
            logger.info(f"Models loaded from {model_dir}")
            tab1, tab2 = st.tabs(["Quick Scenario", "Custom Features"])
            with tab1:
                st.subheader("Predefined Scenarios")
                scenario = st.selectbox("Choose scenario", ["Excellent", "Good", "Poor"])

                # Build domain-aware scenario features (not simply max/min for all features)
                def build_scenario_features(level: str):
                    rng = feature_extractor.feature_ranges
                    names = feature_extractor.get_feature_names()
                    def rmin(n):
                        return float(rng.get(n, (0.0, 1.0))[0])
                    def rmax(n):
                        return float(rng.get(n, (0.0, 1.0))[1])
                    def rmid(n):
                        a, b = rng.get(n, (0.0, 1.0))
                        return float((a + b) / 2)

                    # Defaults
                    vals = {n: rmid(n) for n in names}

                    # Directional adjustments
                    if level == "Excellent":
                        vals['word_count'] = rmax('word_count') * 0.9
                        vals['sentence_count'] = rmax('sentence_count') * 0.8
                        vals['avg_sentence_length'] = max(rmin('avg_sentence_length'), rmid('avg_sentence_length') * 0.8)  # shorter
                        vals['header_count'] = rmax('header_count') * 0.9
                        vals['code_block_count'] = rmid('code_block_count')  # balanced
                        vals['table_count'] = rmax('table_count') * 0.7
                        vals['list_count'] = rmax('list_count') * 0.8
                        vals['security_mentions'] = rmax('security_mentions')
                        vals['scalability_mentions'] = rmax('scalability_mentions')
                        vals['api_mentions'] = rmax('api_mentions')
                        vals['service_count'] = rmax('service_count') * 0.8
                        vals['entity_count'] = rmax('entity_count') * 0.8
                        vals['api_endpoint_count'] = rmax('api_endpoint_count') * 0.8
                        vals['readability_score'] = 95.0
                        vals['documentation_quality'] = 95.0
                        # Filler features neutral-high
                        for i in range(16, 38):
                            vals[f'feature_{i}'] = 80.0
                    elif level == "Good":
                        vals['word_count'] = rmid('word_count') * 1.1
                        vals['sentence_count'] = rmid('sentence_count')
                        vals['avg_sentence_length'] = rmid('avg_sentence_length')
                        vals['header_count'] = rmid('header_count')
                        vals['code_block_count'] = rmid('code_block_count')
                        vals['table_count'] = rmid('table_count')
                        vals['list_count'] = rmid('list_count')
                        vals['security_mentions'] = rmid('security_mentions') * 1.2
                        vals['scalability_mentions'] = rmid('scalability_mentions') * 1.2
                        vals['api_mentions'] = rmid('api_mentions') * 1.2
                        vals['service_count'] = rmid('service_count')
                        vals['entity_count'] = rmid('entity_count')
                        vals['api_endpoint_count'] = rmid('api_endpoint_count')
                        vals['readability_score'] = 80.0
                        vals['documentation_quality'] = 80.0
                        for i in range(16, 38):
                            vals[f'feature_{i}'] = 60.0
                    else:  # Poor
                        vals['word_count'] = rmin('word_count') * 0.8
                        vals['sentence_count'] = rmin('sentence_count') * 0.8
                        vals['avg_sentence_length'] = min(rmax('avg_sentence_length'), rmid('avg_sentence_length') * 1.3)  # longer
                        vals['header_count'] = rmin('header_count')
                        vals['code_block_count'] = rmin('code_block_count')
                        vals['table_count'] = rmin('table_count')
                        vals['list_count'] = rmin('list_count')
                        vals['security_mentions'] = rmin('security_mentions')
                        vals['scalability_mentions'] = rmin('scalability_mentions')
                        vals['api_mentions'] = rmin('api_mentions')
                        vals['service_count'] = rmin('service_count')
                        vals['entity_count'] = rmin('entity_count')
                        vals['api_endpoint_count'] = rmin('api_endpoint_count')
                        vals['readability_score'] = 40.0
                        vals['documentation_quality'] = 40.0
                        for i in range(16, 38):
                            vals[f'feature_{i}'] = 30.0

                    # Clip into ranges
                    for n in names:
                        a, b = rng.get(n, (0.0, 100.0))
                        vals[n] = float(max(a, min(b, vals.get(n, rmid(n)))))
                    return vals

                features = build_scenario_features(scenario)
                if st.button("Predict Scenario Quality"):
                    logger.info(f"Predicting scenario quality for: {scenario}")
                    preds = predictor.predict(features)
                    # Nicely formatted UI
                    def _clip01(v):
                        try:
                            return max(0.0, min(100.0, float(v)))
                        except Exception:
                            return 0.0
                    overall = _clip01(preds.get('ensemble_average', 0.0))
                    st.markdown("### Overall Quality")
                    st.metric("Overall", f"{overall:.1f}/100")
                    try:
                        st.progress(int(round(overall)))
                    except Exception:
                        pass
                    st.markdown("### Model Breakdown")
                    # Show per-model predictions in neat columns
                    items = [(k, v) for k, v in preds.items() if k != 'ensemble_average']
                    if items:
                        cols = st.columns(min(4, len(items)))
                        for idx, (name, val) in enumerate(items):
                            with cols[idx % len(cols)]:
                                score = _clip01(val)
                                st.metric(label=name, value=f"{score:.1f}/100")
                    logger.info(f"Prediction result: {preds}")
            with tab2:
                st.subheader("Custom Features")
                custom_features = {}
                for name in feature_extractor.get_feature_names():
                    rng = feature_extractor.feature_ranges.get(name, (0, 1))
                    custom_features[name] = st.slider(name, float(rng[0]), float(rng[1]), float(rng[0]), help=f"Range: {rng[0]}-{rng[1]}")
                if st.button("Predict Custom Quality"):
                    logger.info(f"Predicting custom quality with features: {custom_features}")
                    preds = predictor.predict(custom_features)
                    def _clip01(v):
                        try:
                            return max(0.0, min(100.0, float(v)))
                        except Exception:
                            return 0.0
                    overall = _clip01(preds.get('ensemble_average', 0.0))
                    st.markdown("### Overall Quality")
                    st.metric("Overall", f"{overall:.1f}/100")
                    try:
                        st.progress(int(round(overall)))
                    except Exception:
                        pass
                    st.markdown("### Model Breakdown")
                    items = [(k, v) for k, v in preds.items() if k != 'ensemble_average']
                    if items:
                        cols = st.columns(min(4, len(items)))
                        for idx, (name, val) in enumerate(items):
                            with cols[idx % len(cols)]:
                                score = _clip01(val)
                                st.metric(label=name, value=f"{score:.1f}/100")
                    logger.info(f"Prediction result: {preds}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("DesignMind GenAI — HLD Generator (local)")

if __name__ == "__main__":
    main()
