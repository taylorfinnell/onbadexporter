struct MyUtils(
    fn ConvertMxsType value type =(
        value as type 
    )
)

MyUtilsInstance = MyUtils()