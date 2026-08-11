"""
Unit testing for AIMIP loader functionality
"""

from code import data
from code.data.model_loader import AIMIP_SCENARIOS
import pytest


def test_aimip_scenario_map():
    assert AIMIP_SCENARIOS['aimip'] == '0K', "aimip should map to 0K."
    assert AIMIP_SCENARIOS['aimip-p2k'] == '2K', "aimip-p2k should map to 2K."
    assert AIMIP_SCENARIOS['aimip-p4k'] == '4K', "aimip-p4k should map to 4K."
    # MD-1p5 uses the no-'p' warming spellings, which must map to the same scenarios.
    assert AIMIP_SCENARIOS['aimip-2k'] == '2K', "aimip-2k should map to 2K."
    assert AIMIP_SCENARIOS['aimip-4k'] == '4K', "aimip-4k should map to 4K."


def test_aimip_loader_invalid_path():
    with pytest.raises(FileNotFoundError, match="Path not found"):
        data.load_aimip('/nonexistent/path')
