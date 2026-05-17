from langchain_core.language_models.chat_models import BaseChatModel
from config import Settings

def get_llm(settings: Settings) -> BaseChatModel:
    """return a langhcain chat model for the configured provider
    all providers implement tehsame basechatmodel interface.
    the rest of the codebase never needs to know which one is active"""
    
    provider = settings.llm_provider
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.openai_model,
                          api_key=settings.openai_api_key,
                          temperature=0,)
        
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.groq_model,
                        api_key=settings.groq_api_key,
                        temperature=0,)
        
    elif provider == "ollama":
        from langchain_community.chat_mdoels import ChatOllama
        return ChatOllama(model=settings.ollama_model,
                         base_url=settings.ollama_base_url,
                         temperature=0,)
        
        
    else:
        raise ValueError(f"Unknown provider : {provider}."
        f"Choose one of 'openai', 'groq', 'ollama' in the .env file")
        
        
def get_embeddings(settings: Settings):
    """return  huggingface sentence transformer embeddings.
        always local, no latency to external service. 
        384 dim vectors fom all-MiniLM-L6-v2 model"""
        
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.embedding_model,
                                    model_kwargs={"device": "cpu"},
                                    encode_kwargs={"normalize_embeddings": True},
                                    )