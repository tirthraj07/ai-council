from src.personality import Personality

def main():
    critic = Personality(
        name="critic",
        system_prompt="You are a critical analyst that challenges assumptions."
    )

    assistant = Personality(
        name="assistant",
        system_prompt="You are a helpful assistant."
    )


if __name__ == "__main__":
    main()
