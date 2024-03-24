from core.common.const import console
from core.memory.schema import SessionLocal, UserSession
from core.memory.function import crud
from core.llm.model import setup


if __name__ == "__main__":
    console.print("[blue]Welcome to local taking llm!")

    user_name = None
    while not user_name:
        user_name = console.input("[blue]Your name: ")

    user_session = crud.create(UserSession(user_name=user_name))
    console.print(f"[cyan]Nice to see you, {user_session.user_name}!")
    gguf_model_path = console.input("[blue]Enter a GGUF model path, "
                                    "this can be a local path or a URL (leave empty for default): ")
    setup(gguf_model_path)
