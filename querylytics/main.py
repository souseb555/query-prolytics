import logging
from shared.infrastructure.agent.main_agent import MainAgent, MainAgentConfig
from shared.infrastructure.agent.special.retrieval_agent import RetrievalAgentConfig
from shared.infrastructure.language_models.openai_gpt import OpenAIGPTConfig

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    try:
        retrieval_config = RetrievalAgentConfig(
            name="RetrievalAgent",
            debug=True,
            llm=OpenAIGPTConfig(
                temperature=0.7
            )
        )

        config = MainAgentConfig(
            name="MainAgent",
            debug=True,
            llm=OpenAIGPTConfig(
                temperature=0.7
            ),
            retrieval_config=retrieval_config
        )
        agent = MainAgent(config)
        
        print("\n=== QueryLytics Demo ===")
        print("QueryLytics: How can I help you? (type 'exit' to quit)")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                print("QueryLytics: Goodbye!")
                break
                
            response = agent.handle_message(user_input)
            print(f"QueryLytics: {response}")
            print(f"[Debug] Current State: {agent.state}")

    except Exception as e:
        logger.error("Error in main: %s", str(e))
        raise

if __name__ == "__main__":
    main() 