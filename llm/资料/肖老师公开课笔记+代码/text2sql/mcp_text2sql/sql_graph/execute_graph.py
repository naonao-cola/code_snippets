import asyncio

from sql_graph.draw_png import draw_graph
from sql_graph.text2sql_graph import make_graph


async def execute_graph():
    """执行该 工作流"""
    async with make_graph() as graph:
        draw_graph(graph, 'text2sql.png')
        while True:
            user_input = input("用户：")
            if user_input.lower() in ['q', 'exit', 'quit']:
                print('对话结束，拜拜！')
                break
            else:
                async for event in graph.astream({"messages": [{"role": "user", "content": user_input}]},
                                                 stream_mode="values"):
                    event["messages"][-1].pretty_print()


if __name__ == '__main__':
    # db = SQLDatabase.from_uri('sqlite:///../chinook.db')
    # toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    #
    # tools = toolkit.get_tools()
    # for tool in tools:
    #     print('工具名字: ', tool.name, '工具描述: ', tool.description)
    #
    # list_tables_tool = next(tool for tool in tools if tool.name == 'sql_db_list_tables')
    #
    # resp = list_tables_tool.invoke("")
    # print(resp)
    asyncio.run(execute_graph())

