from typing import Any, Dict, List, Optional, Union
from apropos.src.core.programs.dag import TextGraphNode
from apropos.src.lms.helpers import LLM
## Graph is Stateful, StatefulGraphNode has read/write methods


class StateCallable:
    pass


class ArbitraryGraphNode:
    gn: TextGraphNode
    read_from_state: StateCallable
    render_output_for_aci: StateCallable
    transform: Union[LLM]
    pass
