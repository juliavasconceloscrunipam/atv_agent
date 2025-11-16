import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from serpapi import GoogleSearch
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict

load_dotenv()


llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0
)


@tool
def google_search(query: str):
    """Busca notícias usando SerpAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    result = GoogleSearch(params).get_dict()
    return result


@tool
def sentiment_analysis(text: str):
    """Classifica o sentimento do texto."""
    prompt = f"""
    Classifique o sentimento do texto abaixo como Positivo, Negativo ou Neutro.
    Explique brevemente a razão.
    Responda APENAS em formato JSON válido com as chaves: "label" e "reason".

    Texto:
    {text}
    """

    resp = llm.invoke(prompt)
    return resp.content


class AgentState(TypedDict):
    query: str
    results: Optional[dict]
    analyzed: Optional[List[Dict]]


def step_search(state: AgentState):
    """Executa a busca no Google."""
    query = state["query"]
    print(f"Executando busca para: '{query}'")
    
    api_key = os.getenv("SERPAPI_API_KEY")
    
    if not api_key:
        print("ERRO: SERPAPI_API_KEY não encontrada no arquivo .env")
        return {"results": {}}
    
    print(f"API Key encontrada: {api_key[:10]}...")
    
    try:
        data = GoogleSearch({
            "engine": "google",
            "q": query,
            "api_key": api_key
        }).get_dict()
        
        if 'error' in data:
            print(f"ERRO DA API SERPAPI: {data['error']}")
            return {"results": {}}
        
        print(f"Busca concluída! Resultados encontrados: {len(data.get('organic_results', []))}")
        
        if data.get('organic_results'):
            print(f"Primeira notícia: {data['organic_results'][0].get('title', 'Sem título')}")
        else:
            print("Nenhum resultado em 'organic_results'")
            print(f"Keys disponíveis no retorno: {list(data.keys())}")
        
        return {"results": data}
    
    except Exception as e:
        print(f"ERRO na busca: {e}")
        return {"results": {}}


def step_analyze(state: AgentState):
    """Analisa o sentimento das notícias."""
    print("\nIniciando análise de sentimento...")
    
    analyzed = []

    results = state.get("results", {})
    news_items = results.get("organic_results", [])
    
    print(f"Total de notícias para analisar: {len(news_items)}")
    
    if not news_items:
        print("AVISO: Nenhuma notícia encontrada para analisar!")
        print(f"Conteúdo de 'results': {list(results.keys()) if results else 'Vazio'}")
        return {"analyzed": []}
    
    for i, item in enumerate(news_items[:5], 1):  # analisa 5 notícias
        title = item.get("title", "")
        if title:
            print(f"\nAnalisando notícia {i}/5: {title[:50]}...")
            
            prompt = f"""
            Classifique o sentimento do texto abaixo como Positivo, Negativo ou Neutro.
            Explique brevemente a razão.
            Responda APENAS em formato JSON válido com as chaves: "label" e "reason".

            Texto:
            {title}
            """
            
            try:
                sent = llm.invoke(prompt)
                analyzed.append({
                    "title": title,
                    "sentiment": sent.content
                })
                print(f"Análise {i} concluída!")
            except Exception as e:
                print(f"Erro ao analisar notícia {i}: {e}")

    print(f"\nAnálise concluída! Total analisado: {len(analyzed)}")
    return {"analyzed": analyzed}


graph = StateGraph(AgentState)

graph.add_node("search", step_search)
graph.add_node("analyze", step_analyze)

graph.set_entry_point("search")
graph.add_edge("search", "analyze")
graph.add_edge("analyze", END)

agent = graph.compile()


if __name__ == "__main__":
    # query = "assassinatos"
    query = "cura câncer descoberta"

    print(f"\nBuscando: {query}\n")
    
    try:
        result = agent.invoke({"query": query})

        print("\n=== RESULTADO FINAL ===")
        if result.get("analyzed"):
            for i, item in enumerate(result["analyzed"], 1):
                print(f"\n{'='*60}")
                print(f"Notícia {i}")
                print(f"{'='*60}")
                print(f"Título: {item['title']}")
                print(f"\nSentimento:\n{item['sentiment']}")
        else:
            print("Nenhum resultado encontrado.")
    
    except Exception as e:
        print(f"\nErro ao executar o agente: {e}")