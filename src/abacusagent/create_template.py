from pathlib import Path
from typing import Union, Optional
import os, sys
from abacusagent.prompt import EXAMPLE_ABACUS_AGENT_INSTRUCTION

def create_google_adk_template(path: Optional[str | Path] = "."):
    tpath = os.path.join(path, "abacus-agent")
    os.makedirs(tpath, exist_ok=True)
    current_file_path = Path(__file__).parent / "google-adk-agent-template.py"
    
    # Read all content in the template file, and replace instructions by the provided example instructions
    with open(current_file_path, 'r', encoding='utf-8') as template_file:
        template_content = template_file.read()
    
    template_content = template_content.replace("{instructions}", '""' + EXAMPLE_ABACUS_AGENT_INSTRUCTION + '""')

    target_file_path = os.path.join(tpath, "agent.py")
    with open(target_file_path, 'w', encoding='utf-8') as target_file:
        target_file.write(template_content)
    
    with open(os.path.join(tpath, "__init__.py"), "w") as file:
        file.write("from . import agent")