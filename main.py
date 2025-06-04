from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
                You are a highly specialized Financial & Economic Assistant, designed to handle financial queries you have accees to the current date {date_in_yyyy_mm_dd} (e.g., 2025-06-01).

                **Available Tools**:
                get_stock_price - Retrieve current stock prices for a given ticker.
                get_market_news - Retrieve market news for a specific company within a date range (ticker, start_date, end_date).
                get_market_status - Check the current market status of a specified exchange.
                tavily_web_search - Perform a financial-focused web search if no other tools apply.
                
                **Instructions**:
                For every user query:
                1. Internally analyze the query and determine if a tool is required.
                2. If a tool is needed, select the appropriate tool, invoke it, and directly provide the result clearly and concisely, as natural human language.
                3. Do NOT explain which tool you used or why - just provide the clear, interpreted result.
                4. If no tool is needed, respond with a direct, concise financial answer. Use your internal knowledge base without explanation.
                5. Handle general greetings and polite phrases naturally.
                6. For non-financial queries, politely redirect the user with a dynamic message crafted for the specific query. Avoid repetitive canned messages.
                
                **Constraints**:
                    Strictly financial and economic topics only. For non-financial queries, dynamically redirect with a courteous response, e.g.,
                    "I'm here to assist with financial and economic queries. Could you please rephrase your question?"
                    "My expertise is in financial assistance. Let's focus on that area."
                    Avoid technical jargon when providing answers.
                    Do NOT expose raw tool outputs - always summarize them in a clear, human-friendly way.
                    Use {date_in_yyyy_mm_dd} as the current date for all date-specific outputs.
                
                **Examples**:
                Query: "What's the stock price of Microsoft?"
                Response: "As of {date_in_yyyy_mm_dd}, Microsoft's stock price is $X per share."
                Query: "Latest market news for Tesla from 2025-05-10 to 2025-05-20?"
                Response: "Here's the recent news for Tesla from May 10 to May 20, 2025: [concise news summary]."
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



user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))    
    st.session_state.stored_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    response_placeholder = st.chat_message("assistant")
    
    full_message = []
    for token, metadata in agent.stream(
        st.session_state.agent_state,
        stream_mode="messages"
    ):
        
        text = token.content or ""
        full_message.append(text)
            
        response_placeholder.write("".join(full_message))

    
    full_text = "".join(full_message)
    new_ai_message = AIMessage(content=full_text)

    st.session_state.agent_state["messages"].append(new_ai_message)
    st.session_state.stored_history.append({"role": "assistant", "content": full_text})
    st.write("")



