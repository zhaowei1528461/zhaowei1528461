
def CodeCheck(code):
    try:
        exec(code)
        success = True
        error_info = ""
    except Exception as e:
        success = False
        error_info = str(e)
    return success,error_info

def LLMCodeDebugger(code,):

    success, error_info=CodeCheck(code)
    print(success, error_info)

LLMCodeDebugger(code)