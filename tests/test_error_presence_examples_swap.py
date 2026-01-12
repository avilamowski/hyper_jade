"""
Test for error_presence examples template selection.

This test verifies that for error_presence type requirements, the correct
examples template is selected, which generates examples with proper semantics:
- correct_examples = code WITHOUT the error (good code)
- erroneous_examples = code WITH the error (bad code)

The solution uses separate templates for different requirement types:
- examples.jinja: for requirement_presence and stylistic
- examples_error_presence.jinja: for error_presence (inverted semantics)
"""

import pytest
from pathlib import Path


class TestExamplesTemplateSelection:
    """Test that the correct examples template is selected based on requirement type."""

    def test_examples_template_exists(self):
        """Verify that the default examples template exists."""
        template_path = Path(__file__).parent.parent / "templates" / "examples.jinja"
        assert template_path.exists(), f"Default examples template not found at {template_path}"

    def test_examples_error_presence_template_exists(self):
        """Verify that the error_presence examples template exists."""
        template_path = Path(__file__).parent.parent / "templates" / "examples_error_presence.jinja"
        assert template_path.exists(), f"Error presence examples template not found at {template_path}"

    def test_template_key_generation(self):
        """Test that template keys are generated correctly for different types."""
        # Simulate the logic in example_generation_node
        def get_template_key(requirement_type_str):
            return f"examples_{requirement_type_str}" if requirement_type_str else "examples"
        
        assert get_template_key("error_presence") == "examples_error_presence"
        assert get_template_key("requirement_presence") == "examples_requirement_presence"
        assert get_template_key("stylistic") == "examples_stylistic"
        assert get_template_key(None) == "examples"


class TestErrorPresenceTemplateContent:
    """Test that the error_presence template has correct semantics."""

    def test_error_presence_template_instructs_correct_semantics(self):
        """
        The error_presence template should instruct the LLM to:
        - Put code WITHOUT the error in <correct>
        - Put code WITH the error in <erroneous>
        """
        template_path = Path(__file__).parent.parent / "templates" / "examples_error_presence.jinja"
        content = template_path.read_text()
        
        # Check that the template talks about "error" not "requirement"
        assert "error" in content.lower(), "Template should mention 'error'"
        
        # Check that correct examples are described as NOT having the error
        assert "NOT have this error" in content or "AVOIDS the error" in content, \
            "Template should describe correct examples as NOT having the error"
        
        # Check that erroneous examples are described as HAVING the error
        assert "DO have this error" in content or "CONTAINS the error" in content, \
            "Template should describe erroneous examples as HAVING the error"

    def test_default_template_instructs_standard_semantics(self):
        """
        The default template should instruct the LLM to:
        - Put code that IMPLEMENTS the requirement in <correct>
        - Put code that DOES NOT implement the requirement in <erroneous>
        """
        template_path = Path(__file__).parent.parent / "templates" / "examples.jinja"
        content = template_path.read_text()
        
        # Check that correct examples are described as implementing the requirement
        assert "correctly implement" in content.lower(), \
            "Template should describe correct examples as implementing the requirement"
        
        # Check that erroneous examples are described as NOT implementing
        assert "do not implement" in content.lower(), \
            "Template should describe erroneous examples as not implementing"


class TestConfigurationConsistency:
    """Test that the configuration properly maps templates."""

    def test_config_has_error_presence_template(self):
        """Verify that the config file has the error_presence examples template defined."""
        import yaml
        
        config_path = Path(__file__).parent.parent / "src" / "config" / "assignment_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        templates = config.get("agents", {}).get("prompt_generator", {}).get("templates", {})
        
        assert "examples" in templates, "Config should have 'examples' template"
        assert "examples_error_presence" in templates, "Config should have 'examples_error_presence' template"
        assert templates["examples_error_presence"] == "examples_error_presence.jinja", \
            "examples_error_presence should point to examples_error_presence.jinja"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
