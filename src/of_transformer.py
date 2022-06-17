'''ORDINAL FEATURE HANDLING''' # use in main()

ordinal_columns_str = ['Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                       'BsmtFinType1', 'BsmtFinType2', 
                       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 
                       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PoolQC', 'Fence', 'MiscFeature']

mappings = [{'AllPub' : 4, 'NoSewr' : 3, 'NoSeWa' : 2, 'ELO' : 1},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1, 'NA' : 0},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1, 'NA' : 0},
            {'Gd' : 4, 'Av' : 3, 'Mn' : 2, 'No' : 1, 'NA' : 0},
            {'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'NA' : 0},
            {'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'NA' : 0},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1},
            {'Y' : 1, 'N' : 0},
            {'SBrkr' : 5, 'FuseA' : 3, 'FuseF' : 2, 'FuseP' : 1, 'Mix' : 4},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1},
            {'Typ' : 7, 'Min1' : 5, 'Min2' : 5, 'Mod' : 4, 'Maj1' : 2, 'Maj2' : 2, 'Sev' : 1, 'Sal' : 0},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1, 'NA' : 0},
            {'2Types' : 6, 'Attchd' : 5, 'Basment' : 4, 'BuiltIn' : 3, 'CarPort' : 2, 'Detchd' : 1, 'NA' : 0},
            {'Fin' : 3, 'RFn' : 2, 'Unf' : 1, 'NA' : 0},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1, 'NA' : 0},
            {'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa' : 2, 'Po' : 1, 'NA' : 0},
            {'Ex' : 4, 'Gd' : 3, 'TA': 2, 'Fa' : 1, 'NA' : 0},
            {'GdPrv' : 2, 'MnPrv' : 1, 'GdWo' : 2, 'MnWw' : 1, 'NA' : 0},
            {'Elev' : 1, 'Gar2' : 1, 'Othr' : 1, 'Shed' : 1, 'TenC' : 1, 'NA' : 0}]
            

def ordinal_transformer(df):
    for i in range(len(ordinal_columns_str)):
        df[ordinal_columns_str[i]] = df[ordinal_columns_str[i]].map(mappings[i])

    return df
