def if_native_model(a):
    return select({
        ":ds_native_model": a,
        "//conditions:default": []
    })
