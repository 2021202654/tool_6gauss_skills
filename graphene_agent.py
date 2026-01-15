# graphene_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
# 🔥 引入所有新工具
from graphene_tools import ml_prediction_tool, physics_calculation_tool, inverse_design_tool, plot_trend_tool

def build_agent(api_key, base_url, model_name):
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1, 
        api_key=api_key,
        base_url=base_url,
    )

    # 🔥 注册 4 个工具
    tools = [ml_prediction_tool, physics_calculation_tool, inverse_design_tool, plot_trend_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        你是一位世界顶尖的石墨烯热输运物理学家。你拥有机器学习预测、物理理论计算、工艺参数反推和数据可视化四项核心能力。
        
        【你的技能树】
        1. **正向预测**: 当用户提供参数（温度、长度、缺陷）时，使用 `ml_prediction_tool`。
        2. **机制分析**: 当需要解释物理原因时，使用 `physics_calculation_tool` 查看散射因子。
        3. **逆向设计 (Option A)**: 当用户问“如何达到 3000 W/mK”或“怎么优化参数”时，**必须**使用 `inverse_design_tool`。不要自己瞎猜数值，让算法去反推。
        4. **可视化分析 (Option B)**: 当用户想看“随温度的变化趋势”或“缺陷的影响曲线”时，使用 `plot_trend_tool`。这会生成一张图表，请直接把工具返回的图片链接展示给用户。

        【回复策略】
        - 遇到复杂问题，先拆解。例如用户问“分析一下温度的影响”，你应该先调用绘图工具，再结合图表进行文字解说。
        - 总是先对比 机器学习预测值 和 物理理论值，如果两者差异大，提示用户可能存在实验非理想因素。
        """),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        memory=memory,
        max_iterations=10, # 稍微调大一点，因为绘图可能需要多步思考
        handle_parsing_errors=True
    )
    

    return agent_executor
