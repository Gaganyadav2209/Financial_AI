from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
import finnhub
import os 
from dotenv import load_dotenv
from datetime import date,datetime
import random
import string
from typing import Literal, List
import streamlit as st


load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
tavilysearch_api_key = os.environ.get("TAVILYSEARCH_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")


llm  = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20",google_api_key=gemini_api_key)
alpha_vantage_client = AlphaVantageAPIWrapper(alphavantage_api_key=alpha_vantage_api_key)
finnhub_client = finnhub.Client(api_key=finnhub_api_key)
tavily_search_tool = TavilySearch(
    tavily_api_key = tavilysearch_api_key ,
    max_results = 4,
    topic = "finance",
)


# Schema Definitions

class State(TypedDict):
    system: SystemMessage
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]
    

class StockPriceSearch(BaseModel):
    ticker: str = Field(description="The ticker symbol of the stock to search for")

class MarketNewsSearch(BaseModel):
    ticker: str = Field(description="The ticker symbol of the stock to search for")
    start_date: str = Field(description="The start date from when to fetch the market news, format is YYYY-MM-DD")
    end_date: str = Field(description="The end date till when to fetch the market news, format is YYYY-MM-DD")

class MarketStatusSearch(BaseModel):
    exchange: str = Field(description="The exchange to check the market status for, (e.g. US, UK, etc.)")

class MarketHolidaysSearch(BaseModel):
    exchange: str = Field(description="The exchange to check the market holidays for, (e.g. US, UK, etc.)")



WINDOW_SIZE = 10

def generate_random_id():
    characters = string.ascii_letters + string.digits   
    id_length = 10
    random_id = ''.join(random.choice(characters) for _ in range(id_length))
    return random_id


# Get Stock Price
@tool("get_stock_price",args_schema=StockPriceSearch,return_direct = False)
def get_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a given ticker symbol."""
    try:
        data = alpha_vantage_client._get_quote_endpoint(symbol=ticker)   
        return f"The current price of {ticker} is ${data['Global Quote']['05. price']}"
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"



# Get Market News
@tool("get_market_news",args_schema=MarketNewsSearch,return_direct = False)
def get_market_news(ticker: str, start_date: str, end_date: str) -> str:
    """
    Fetch market news for a given ticker symbol from start_date to end_date.
    Dates should be in YYYY-MM-DD format or 'today' for the current date.
    """
    if not ticker or not start_date or not end_date:
        return "Error: Missing required parameters. Please provide ticker, start_date, and end_date."

    if end_date.lower() == "today":
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Error: Dates must be in YYYY-MM-DD format (e.g., 2025-05-10)."

    if start_date > end_date:
        return "Error: 'From' date cannot be later than 'To' date."

    try:
        news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        if not news:
            return f"No market news found for {ticker} between {start_date} and {end_date}."
        
        limited_news = news[:3]
        formatted_news = []
        for item in limited_news:
            headline = item.get("headline", "No headline")
            summary = item.get("summary", "No summary")
            url = item.get("url", "No URL")
            formatted_news.append(f"- {headline}\nSummary: {summary}\nLink: {url}")
        
        return f"Top {len(limited_news)} market news items for {ticker} from {start_date} to {end_date}:\n\n" + "\n\n".join(formatted_news)
    
    except Exception as e:
        return f"Error fetching news: {str(e)}"



# Get Market Status
@tool("get_market_status",args_schema=MarketStatusSearch,return_direct = False)
def get_market_status(exchange: str) -> str:
    """
    Fetch the current market status for a given exchange.
    """
    try:
        status = finnhub_client.market_status(exchange = exchange)
        
        return f"The current market status for {exchange} is: holiday {status['holiday']}, open: {status['isOpen']}, session: {status['session']} and timezone: {status['timezone']}"

    except Exception as e:
        return f"Error fetching market status: {str(e)}"



# Get Market Holidays
@tool("get_market_holidays",args_schema=MarketHolidaysSearch,return_direct = False)
def get_market_holidays(exchange: str) -> str:
    """
      Fetch the current market holidays for a given exchange.
    """
    try:
        holidays = finnhub_client.market_holiday(exchange = exchange)
        return f"The current market holidays for {exchange} are: {holidays}"
    except Exception as e:
        return f"Error fetching market holidays: {str(e)}"



# Tavily Web Search
@tool("tavily_web_search", return_direct=False)
def tavily_web_search(query: str) -> str:
    """
    Perform a finance-focused web search for `query` and return top 3 snippet contents as a single 
    bullet-list string. Uses a hypothetical `tavily_search_tool` under the hood.
    """
    try:
        results = tavily_search_tool.invoke(query)
        response_dict = {}
        for item in results.get("results", [])[:3]:
            content = item.get("content","")
            url = item.get("url","")
            response_dict[url] = content
        
        if not response_dict:
            return "No relevant web results found."
      
        return f"Top web search results for {query} are: {response_dict}"

    except Exception as e:
        return f"Error in web search: {str(e)}"

    

tools = [get_stock_price,get_market_news,get_market_status,get_market_holidays,tavily_web_search]

llm_with_tools = llm.bind_tools(tools)


def call_tools(state: State) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: State):
    messages = state["messages"]
    system_msg = state["system"]
    recent_messages = messages[-WINDOW_SIZE:] if len(messages) > WINDOW_SIZE else messages
    to_send = [system_msg] + recent_messages
    
    response = llm_with_tools.invoke(to_send)
    return {"messages": [response]}



tool_node = ToolNode(tools=tools)



workflow = StateGraph(State)
workflow.add_node("LLM", call_model)
workflow.add_edge(START, "LLM")
workflow.add_node("tools", tool_node)
workflow.add_conditional_edges("LLM", call_tools)
workflow.add_edge("tools", "LLM")

agent = workflow.compile()

st.title("AI Powered Financial Assistant")
st.write("Ask me anything about the financial market")


#Current date
now = datetime.now()
date_in_yyyy_mm_dd = now.strftime("%Y-%m-%d")

initial_state: State = {
    "system": SystemMessage(
        content=(
            """
                You are a highly specialized Financial & Economic Assistant, designed to help users with tasks related strictly to finance, stock markets, economic indicators, and financial news. You operate with access to a set of tools and always reference the current date: {date_in_yyyy_mm_dd}.

                ⚠️ Always use {date_in_yyyy_mm_dd} as your point of reference — never default to your model's internal knowledge date. Ignore knowledge cut-off unless answering historical questions.
                🧠 Core Behavior:
                Internally evaluate the user query to determine intent and whether a tool is required.
                If a tool is required, use it silently. Do not tell the user which tool or parameters were used.
                If no tool is needed, use your internal knowledge and return a direct answer.
                If a web search is used, return a rich, structured summary with clear citations/URLs where relevant.
                Do not expose or explain internal operations, tool choices, or system reasoning in your reply.
                Avoid technical jargon — prefer clarity and accessibility.
                🔒 Hard Constraints (Must Always Follow):
                Only answer financial and economic questions. Do not respond to unrelated topics — even if asked repeatedly.
                If the query is out of scope, gently guide the user back to finance/economics in a personalized, natural way.
                Always use {date_in_yyyy_mm_dd} as the current date in all time-sensitive responses.
                Never reveal tool names, parameters, or internal logic to the user.
                Never state or imply you're limited by knowledge cutoff if {date_in_yyyy_mm_dd} is provided.
                Handle general greetings (e.g. “Hi”, “Thanks”) with polite, concise responses.
                🛠️ Available Internal Tools (Invisible to User):
                get_stock_price: Get live stock prices.
                get_market_news: Get company-specific market news within a date range.
                get_market_status: Check open/closed status of global stock exchanges.
                tavily_web_search: Perform a financial-specific search when no other tools apply.
                These tools are internal resources — do not refer to them in any user response.

                📚 Examples (Response Structure & Tone):
                Query: "What's the stock price of Microsoft today?"
                Response: "As of {date_in_yyyy_mm_dd}, Microsoft's stock is trading at $X per share."

                Query: "Market news for Tesla from May 15 to May 20?"
                Response: "Here's a summary of the recent Tesla news between May 15 and May 20:

                Tesla announced updates to its full self-driving software.
                Production at its Berlin Gigafactory exceeded targets.
                [More insights as needed]"
                Query: "Hi!"
                Response: "Hello! I'm here to help with financial and economic questions. What would you like to know today?"

                Query: "What's your favorite movie?"
                Response: "I specialize in finance and economic topics. Feel free to ask me about stocks, markets, or financial trends!"

                🧩 Best Practices During Execution:
                Think step-by-step internally to identify whether a query is about:
                A real-time stock price → Use internal price tool.
                Company news with dates → Use market news tool.
                Market hours → Use market status tool.
                General but current finance-related query → Use web search.
                Knowledgeable question that doesn't require fresh data → Use your internal finance/economic knowledge.
            """
        )
    ),
    "messages": [],
   
}


if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "system": initial_state["system"],
        "messages": []
    }


if "stored_history" not in st.session_state:
    st.session_state.stored_history = []



for entry in st.session_state.stored_history:
    role = entry["role"]
    content = entry["content"]
    if role == "user":
        st.chat_message("user").write(content)
    elif role == "assistant":
        st.chat_message("assistant").write(content)




user_input = st.chat_input("Type your message here…")
if user_input:   
    st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))
    st.session_state.stored_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

   
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_parts = []
        for token, metadata in agent.stream(
            st.session_state.agent_state,
            stream_mode="messages"
        ):
            if isinstance(token, ToolMessage):
                continue
            chunk = token.content or ""
            full_parts.append(chunk)
            current_text = "".join(full_parts)
          
            placeholder.write(current_text)

    full_text = "".join(full_parts)
    ai_msg = AIMessage(content=full_text)
    st.session_state.agent_state["messages"].append(ai_msg)
    st.session_state.stored_history.append({"role": "assistant", "content": full_text})
