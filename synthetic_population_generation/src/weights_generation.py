


def compute_weights(METADATA_PATH):

    METADATA_FILE = "FADN_metadata_spain.xlsx"

    metadata = pd.read_excel(os.path.join(METADATA_PATH, METADATA_FILE))

    metadata["(SYS02) Farms represented (nb)"] = metadata["(SYS02) Farms represented (nb)"].astype(int)

    es_naming = {'(2) 8 000 - < 25 000 EUR': 1, 
                '(3) 25 000 - < 50 000 EUR': 2,
                '(4) 50 000 - < 100 000 EUR': 3, 
                '(5) 100 000 - < 500 000 EUR': 4,
                '(6) >= 500 000 EUR': 5}

    metadata["Economic Size"] = metadata["Economic Size"].apply(lambda x: es_naming[x])

    metadata["(SE005) Economic size (€'000)"] = metadata["(SE005) Economic size (€'000)"].apply(lambda x: 1000*x)
    microdata["SE005"] = microdata["SE005"].apply(lambda x: 1000*x)

    metadata = metadata.rename(columns={"(SE005) Economic size (€'000)": "(SE005) Economic size (€)"})

    NUTS2_conversion = {'(575) Andalucía': "ES61"}

    metadata["Region"] = metadata["Region"].apply(lambda x: NUTS2_conversion[x])
    metadata = metadata.rename(columns={"Region": "A_LO_40_N2", 
                                        "Year": "YEAR"})
                                        
    metadata.head(10)