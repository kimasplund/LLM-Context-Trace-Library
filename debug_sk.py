try:
    import semantic_kernel as sk

    print(f"Semantic Kernel version: {sk.__version__}")
    from semantic_kernel.filters import FunctionInvocationContext, FilterReturnType

    try:
        from semantic_kernel.filters import FunctionInvocationFilter

        print("FunctionInvocationFilter: OK")
    except ImportError:
        print("FunctionInvocationFilter: Missing")
        # Try finding AutoFunctionInvocationFilter or similar
        try:
            from semantic_kernel.filters import AutoFunctionInvocationFilter

            print("AutoFunctionInvocationFilter: OK")
        except ImportError:
            print("AutoFunctionInvocationFilter: Missing")

except Exception as e:
    print(f"Error: {e}")
