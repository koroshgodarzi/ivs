import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from reportingAssistant.schema import GraphState
import os
from langchain_core.runnables import RunnableConfig
import json
from utils import get_llm, extract_json_from_text, ommiting_think_block

def visualization(state: GraphState, config: RunnableConfig) -> dict:
    """
    Prompts the LLM to determine the best chart_type and parameters based on the 
    user's question and the retrieved SQL data.
    """
    # Extract user question
    user_messages = [msg for msg in state.get("messages", []) if msg.get("role") == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    # Extract query results
    query_results_str = state.get("query_results", "[]")
    
    # Optional: Truncate query results to avoid exceeding token limits
    # We only need to show the LLM the schema and a few sample rows to decide the chart
    try:
        # results_list = json.loads(query_results_str)
        # # for i in range(5):
        # print(type(results_list))
        # print(results_list.items())
        sample_data = query_results_str[:5] if isinstance(query_results_str, list) else query_results_str
        sample_data_str = json.dumps(sample_data, indent=2)
    except json.JSONDecodeError:
        sample_data_str = query_results_str

    # Retrieve parameters from the config
    configurable = config.get("configurable", {})
    prompt_dir = configurable.get("prompt_template_dir", os.path.join("..", "prompt_template"))
    model_name = configurable.get("model_name", "gpt")
    
    llm = get_llm(model_name)

    # Load prompt template
    prompt_template_path = os.path.join(prompt_dir, 'visualization_prompt.txt')
    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template not found at {prompt_template_path}")
        return {"error_message": state.get("error_message", []) + ["Visualization prompt missing."]}

    # Format the prompt
    prompt = prompt.replace("[USER_QUESTION_HERE]", user_question)
    prompt = prompt.replace("[QUERY_RESULTS_HERE]", sample_data_str)
    
    # Invoke the LLM
    response = llm.invoke(prompt)
    
    # Clean and extract JSON (using your provided util functions)
    try:
        content = response.content
    
        if isinstance(content, str):
            response_content = content.strip()
        elif isinstance(content, list):
            # Join text fields from all content blocks in the list
            response_content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            response_content = ""
        response_content = ommiting_think_block(response_content)        
        viz_config = extract_json_from_text(response_content)
    except Exception as e:
        print(f"Error extracting JSON from LLM: {e}")
        viz_config = {"chart_type": None, "params": {}}

    # Return the updates for GraphState. 
    # Note: Ensure you add 'viz_config' or 'chart_type'/'chart_params' to your GraphState TypedDict!
    return {
        "viz_config": viz_config
    }


def rendering_node(state: GraphState, config: RunnableConfig):
    viz_config = state.get("viz_config", {})
    chart_type = viz_config.get("chart_type")
    params = viz_config.get("params", {})
    
    # Construct the JSON data format expected by your plotting function
    print()
    print(state["query_results"])
    json_data = {"query_results": state["query_results"]}
    
    # Generate the chart
    fig = plotting(json_data, chart_type, params)
    
    # You can now save `fig` to a file, display it via Streamlit/Dash, etc.
    if hasattr(fig, 'show'):
        fig.show()
        
    return state


def plotting(json_data, chart_type, params):
    """
    Generates a chart based on SQL query results and PM requirements.
    
    :param json_data: dict, The raw JSON containing 'query_results'
    :param chart_type: str, e.g., 'bar', 'line', 'pie', 'gantt', 'heatmap', 'scatter'
    :param params: dict, Mapping of chart components to column names 
                         (e.g., {'x': 'ProjectName', 'y': 'Budget'})
    :return: A Plotly Figure object
    """
    # 1. Convert JSON to DataFrame
    df = pd.DataFrame(json_data['query_results'])
    
    if df.empty:
        return "No data available to plot."

    # 2. Logic for every feasible PM chart
    try:
        if chart_type == 'bar':
            # Use: Comparing Budget vs Actual, Task counts by User, etc.
            fig = px.bar(df, 
                         x=params.get('x'), 
                         y=params.get('y'), 
                         color=params.get('color'),
                         barmode='group',
                         title=params.get('title', "Comparison Chart"))

        elif chart_type == 'line':
            # Use: Burn-down charts, Spending trends over time.
            # Ensure date columns are datetime objects
            if 'x' in params:
                df[params['x']] = pd.to_datetime(df[params['x']])
            fig = px.line(df, 
                          x=params.get('x'), 
                          y=params.get('y'), 
                          color=params.get('color'),
                          title=params.get('title', "Trend Analysis"))

        elif chart_type == 'pie':
            # Use: Task status distribution (e.g., % Done, % In Progress).
            fig = px.pie(df, 
                         values=params.get('values'), 
                         names=params.get('names'), 
                         hole=0.4, # Makes it a Donut chart (cleaner look)
                         title=params.get('title', "Distribution"))

        elif chart_type == 'gantt':
            # Use: Project Timelines. 
            # Requires Start Date, End Date, and Task Name.
            df[params['start']] = pd.to_datetime(df[params['start']])
            df[params['end']] = pd.to_datetime(df[params['end']])
            fig = px.timeline(df, 
                              x_start=params.get('start'), 
                              x_end=params.get('end'), 
                              y=params.get('task'), 
                              color=params.get('color'),
                              title=params.get('title', "Project Timeline"))
            fig.update_yaxes(autorange="reversed") # Critical for Gantt readability

        elif chart_type == 'scatter':
            # Use: Risk Assessment (Probability vs Impact).
            fig = px.scatter(df, 
                             x=params.get('x'), 
                             y=params.get('y'), 
                             size=params.get('size'), 
                             color=params.get('color'),
                             hover_name=params.get('hover_label'),
                             title=params.get('title', "Risk/Correlation Map"))

        elif chart_type == 'heatmap':
            # Use: Resource Overload or Risk Matrix.
            # Expects a pivot-like structure
            fig = px.density_heatmap(df, 
                                     x=params.get('x'), 
                                     y=params.get('y'), 
                                     z=params.get('z'),
                                     title=params.get('title', "Resource/Risk Heatmap"))

        elif chart_type == 'funnel':
            # Use: Sales Pipeline or Project Stage conversion.
            fig = px.funnel(df, 
                            x=params.get('x'), 
                            y=params.get('y'),
                            title=params.get('title', "Project Pipeline"))

        elif chart_type == 'indicator':
            # Use: High-level KPIs (Total Budget, Overall % Complete).
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = df[params['value']].iloc[0],
                title = {'text': params.get('title')},
                domain = {'x': [0, 1], 'y': [0, 1]}
            ))

        else:
            return f"Chart type '{chart_type}' is not supported."

        # Enhance layout for professional look
        fig.update_layout(template="plotly_white", title_x=0.5)
        return fig

    except Exception as e:
        return f"Error generating chart: {str(e)}"