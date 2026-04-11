from core.config import SEARCH_MODEL


class WebReasoner:
    """Thin wrapper around AutoResearchWorkflow.

    model_name is a class attribute so build_runtime_profile() can read it
    without constructing a live instance (which would pull in SearchHelper and
    potentially trigger model loads).
    """

    model_name = SEARCH_MODEL  # class-level — readable without instantiation

    def __init__(self, search_tool):
        # Import deferred to here so that merely importing web_reasoner at the
        # module level does not construct AutoResearchWorkflow / SearchHelper.
        from reasoning.autoresearch import AutoResearchWorkflow  # noqa: PLC0415
        self._search_tool = search_tool
        self.workflow = AutoResearchWorkflow(search_tool)

    def run(self, query: str) -> dict:
        return self.workflow.run(query)
