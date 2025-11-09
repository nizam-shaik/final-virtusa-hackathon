from __future__ import annotations
"""
Behavior and Quality Node - Analyzes use cases, NFRs, and risks
"""

# TODO: Implement BehaviorQualityNode extending BaseNode
# TODO: Import BehaviorQualityAgent from agent module
# TODO: Implement __init__ to initialize with agent instance
# TODO: Implement execute(state) method:
#       - Create BehaviorQualityAgent instance
#       - Call agent.process(state)
#       - Validate behavior data (use cases, NFRs, risks)
#       - Update state.behavior with BehaviorData
#       - Update stage status to completed
#       - Return updated state
# TODO: Add use case validation
#       - Ensure use cases are clear and concise
#       - Check for actor/action/outcome structure
#       - Remove duplicates
# TODO: Implement NFR validation
#       - Verify all categories present (security, reliability, etc.)
#       - Check NFRs are measurable and testable
#       - Validate category names are standard
# TODO: Implement risk assessment
#       - Validate risk IDs are unique
#       - Check impact and likelihood scores in 1-5 range
#       - Ensure mitigation strategies are defined
#       - Link risks to security threats if applicable
# TODO: Diagram plan processing
#       - Extract actors from use cases
#       - Define message flows between actors
#       - Create sequence diagram specifications
# TODO: Log quality metrics
#       - Use case count
#       - NFR categories and item counts
#       - Risk count and average severity
"""
Behavior and Quality Node - Analyzes use cases, NFRs, and risks
"""

import logging
import statistics
from typing import List, Dict, Any

from .base_node import BaseNode
from agent.behavior_agent import BehaviorQualityAgent
from state.models import HLDState, BehaviorData, RiskData

logger = logging.getLogger(__name__)


class BehaviorQualityNode(BaseNode):
    """
    Node responsible for analyzing system behavior, non-functional requirements (NFRs),
    and risk assessment based on requirement documents.
    """

    def __init__(self):
        super().__init__(name="behavior_quality", agent=BehaviorQualityAgent())

    def execute(self, state: HLDState) -> HLDState:
        """
        Execute BehaviorQualityAgent and update state.behavior with validated, structured data.
        """
        logger.info("[BehaviorQualityNode] Starting behavior and quality analysis.")
        state.set_status("behavior_quality", "processing", "Analyzing use cases and quality metrics")

        try:
            # Run the BehaviorQualityAgent
            result = self.agent.process(state)
            if not isinstance(result, dict):
                raise ValueError("BehaviorQualityAgent did not return a valid result dict")

            # Extract fields
            use_cases = result.get("use_cases", [])
            nfrs = result.get("nfrs", {})
            risks = result.get("risks", [])
            diagram_plan = result.get("diagram_plan", {})

            # --- Use Case Validation ---
            validated_use_cases: List[str] = []
            for uc in use_cases:
                if not isinstance(uc, str):
                    continue
                uc_clean = uc.strip()
                if not uc_clean:
                    continue
                # Expect simple actor-action-outcome format
                if "as a" in uc_clean.lower() or "i want" in uc_clean.lower():
                    validated_use_cases.append(uc_clean)
                else:
                    # Accept general case but log warning
                    logger.warning(f"[BehaviorQualityNode] Use case lacks actor/action form: '{uc_clean}'")
                    validated_use_cases.append(uc_clean)

            # Remove duplicates
            validated_use_cases = list(dict.fromkeys(validated_use_cases))

            if not validated_use_cases:
                validated_use_cases = ["As a user, I want to perform system actions effectively."]
                logger.warning("[BehaviorQualityNode] No valid use cases found; added default placeholder.")

            # --- NFR Validation ---
            standard_categories = {"security", "reliability", "performance", "operability"}
            validated_nfrs: Dict[str, List[str]] = {}
            for cat, items in (nfrs or {}).items():
                cname = cat.strip().lower()
                if cname not in standard_categories:
                    logger.warning(f"[BehaviorQualityNode] Non-standard NFR category detected: '{cat}'")
                validated_nfrs[cname] = [i.strip() for i in (items or []) if i.strip()]

            # Ensure all categories exist
            for cat in standard_categories:
                if cat not in validated_nfrs:
                    validated_nfrs[cat] = []
                    logger.warning(f"[BehaviorQualityNode] Missing NFR category '{cat}', initialized empty.")

            # --- Risk Assessment Validation ---
            validated_risks: List[RiskData] = []
            seen_ids = set()
            for r in risks:
                try:
                    rid = str(r.get("id") or f"RISK-{len(validated_risks)+1}").strip()
                    if rid in seen_ids:
                        rid = f"{rid}-{len(seen_ids)+1}"
                    seen_ids.add(rid)

                    desc = str(r.get("desc") or "No description provided").strip()
                    mitigation = str(r.get("mitigation") or "Mitigation not specified").strip()
                    assumption = str(r.get("assumption") or "").strip()
                    impact = int(r.get("impact") or 3)
                    likelihood = int(r.get("likelihood") or 3)

                    # Ensure scores in valid range 1–5
                    if impact < 1 or impact > 5:
                        impact = 3
                        logger.warning(f"[BehaviorQualityNode] Invalid impact score for {rid}, defaulted to 3.")
                    if likelihood < 1 or likelihood > 5:
                        likelihood = 3
                        logger.warning(f"[BehaviorQualityNode] Invalid likelihood score for {rid}, defaulted to 3.")

                    # Link risks to known security threats
                    linked_threats = []
                    if state.authentication and state.authentication.threats:
                        for t in state.authentication.threats:
                            if any(word in desc.lower() for word in t.lower().split()):
                                linked_threats.append(t)

                    if linked_threats:
                        desc += f" (Related threats: {', '.join(linked_threats)})"

                    validated_risks.append(
                        RiskData(
                            id=rid,
                            desc=desc,
                            assumption=assumption,
                            mitigation=mitigation,
                            impact=impact,
                            likelihood=likelihood,
                        )
                    )
                except Exception as ex:
                    logger.warning(f"[BehaviorQualityNode] Skipped invalid risk: {ex}")

            if not validated_risks:
                validated_risks.append(
                    RiskData(
                        id="RISK-DEFAULT",
                        desc="Potential system reliability or compliance issue.",
                        assumption="Operational risk under load conditions.",
                        mitigation="Add monitoring and fallback mechanisms.",
                        impact=3,
                        likelihood=3,
                    )
                )
                logger.warning("[BehaviorQualityNode] No valid risks found; added default risk entry.")

            # --- Sequence Diagram Generation ---
            # Extract actors from use cases
            actors = set()
            for uc in validated_use_cases:
                if "as a" in uc.lower():
                    parts = uc.lower().split("as a", 1)[-1].split(",")[0].strip().title()
                    actors.add(parts or "User")
                else:
                    actors.add("User")

            # Generate simple sequence diagram structure
            sequences = []
            for idx, uc in enumerate(validated_use_cases):
                steps = [{"from": list(actors)[0], "to": "System", "message": uc}]
                sequences.append({
                    "title": f"Use Case #{idx+1}",
                    "actors": list(actors),
                    "steps": steps
                })

            diagram_plan = diagram_plan or {}
            diagram_plan["sequences"] = sequences

            # --- Log Quality Metrics ---
            avg_severity = round(statistics.mean([(r.impact + r.likelihood) / 2 for r in validated_risks]), 2)
            logger.info(f"[BehaviorQualityNode] Use cases: {len(validated_use_cases)}")
            logger.info(f"[BehaviorQualityNode] NFR categories: {len(validated_nfrs)}")
            logger.info(f"[BehaviorQualityNode] Risks: {len(validated_risks)}, Avg severity: {avg_severity}")

            # --- Update State ---
            behavior_data = BehaviorData(
                use_cases=validated_use_cases,
                nfrs=validated_nfrs,
                risks=validated_risks,
                diagram_plan=diagram_plan
            )

            state.behavior = behavior_data
            state.set_status("behavior_quality", "completed", "Behavior and quality analysis completed")
            return state

        except Exception as e:
            logger.exception("[BehaviorQualityNode] Behavior and quality analysis failed.")
            state.add_error(str(e))
            state.set_status("behavior_quality", "failed", str(e))
            return state
