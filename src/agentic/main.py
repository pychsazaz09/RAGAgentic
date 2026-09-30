if __package__ in (None, ""):
    # Direct script execution needs the src directory on the import path.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agentic.agent import Agent
else:
    from .agent import Agent
    
def main():
    print("Hello from 03-agentic-rag!")
    agent=Agent()
    agent.invoke("你好")



if __name__ == "__main__":
    main()
