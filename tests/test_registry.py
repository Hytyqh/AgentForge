from agentforge.tools.registry import ToolRegistry


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def add(a: int, b: int) -> int:
        """把两个整数相加。

        Args:
            a: 第一个加数
            b: 第二个加数
        """
        return a + b

    @reg.tool(description="手写描述覆盖 docstring")
    def noop() -> str:
        return "ok"

    return reg


def test_schema_generation():
    reg = make_registry()
    schemas = {s["function"]["name"]: s["function"] for s in reg.schemas()}
    add = schemas["add"]
    assert add["description"] == "把两个整数相加。"
    assert add["parameters"]["properties"]["a"]["type"] == "integer"
    assert add["parameters"]["properties"]["a"]["description"] == "第一个加数"
    assert add["parameters"]["required"] == ["a", "b"]
    assert schemas["noop"]["description"] == "手写描述覆盖 docstring"


def test_execute_success():
    reg = make_registry()
    assert reg.execute("add", '{"a": 1, "b": 2}') == "3"


def test_unknown_tool():
    reg = make_registry()
    assert "[工具不存在]" in reg.execute("nope", "{}")


def test_bad_json():
    reg = make_registry()
    assert "[参数解析失败]" in reg.execute("add", "{oops")


def test_wrong_args():
    reg = make_registry()
    assert "[参数不匹配]" in reg.execute("add", '{"a": 1}')


def test_calculate_rejects_code_injection():
    from agentforge.tools.builtin import registry as builtin

    result = builtin.execute("calculate", '{"expression": "__import__(\'os\').getcwd()"}')
    assert result.startswith("[工具执行失败]") or result.startswith("[参数")


def test_calculate_ok():
    from agentforge.tools.builtin import registry as builtin

    assert builtin.execute("calculate", '{"expression": "(3+5)*2**3"}') == "64"


def test_truncation():
    reg = ToolRegistry()

    @reg.tool
    def big() -> str:
        """返回超长文本。"""
        return "x" * 5000

    out = reg.execute("big", "")
    assert len(out) < 5000
    assert "已截断" in out
