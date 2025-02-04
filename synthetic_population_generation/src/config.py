import pandas as pd


totals_variables = ["SE005", 
                    "SE010", 
                    "SE011", 
                    "SE015", 
                    "SE016", 
                    "SE020", 
                    "SE021", 
                    "SE025", 
                    "SE026", 
                    "SE027", 
                    "SE028", 
                    "SE030", 
                    "SE035", 
                    "SE041", 
                    "SE042", 
                    "SE046", 
                    "SE050", 
                    "SE054", 
                    "SE055", 
                    "SE060", 
                    "SE065", 
                    "SE071", 
                    "SE072", 
                    "SE073", 
                    "SE074", 
                    "SE075", 
                    "SE080", 
                    "SE085", 
                    "SE086", 
                    "SE087", 
                    "SE090", 
                    "SE095", 
                    "SE100", 
                    "SE105", 
                    "SE110", 
                    "SE115", 
                    "SE120", 
                    "SE125", 
                    "SE126", 
                    "SE127", 
                    "SE131", 
                    "SE132", 
                    "SE135", 
                    "SE136", 
                    "SE140", 
                    "SE145", 
                    "SE146", 
                    "SE150", 
                    "SE155", 
                    "SE160", 
                    "SE165", 
                    "SE170", 
                    "SE175", 
                    "SE180", 
                    "SE185", 
                    "SE190", 
                    "SE195", 
                    "SE200", 
                    "SE206", 
                    "SE207", 
                    "SE211", 
                    "SE216", 
                    "SE220", 
                    "SE225", 
                    "SE022", 
                    "SE230", 
                    "SE235", 
                    "SE240", 
                    "SE245", 
                    "SE251", 
                    "SE256", 
                    "SE260", 
                    "SE265", 
                    "SE270", 
                    "SE275", 
                    "SE281", 
                    "SE284", 
                    "SE285", 
                    "SE290", 
                    "SE295", 
                    "SE296", 
                    "SE297", 
                    "SE298", 
                    "SE300", 
                    "SE305", 
                    "SE309", 
                    "SE310", 
                    "SE315", 
                    "SE320", 
                    "SE325", 
                    "SE330", 
                    "SE331", 
                    "SE336", 
                    "SE340", 
                    "SE345", 
                    "SE350", 
                    "SE356", 
                    "SE360", 
                    "SE365", 
                    "SE370", 
                    "SE375", 
                    "SE380", 
                    "SE381", 
                    "SE390", 
                    "SE395", 
                    "SE405", 
                    "SE406", 
                    "SE407", 
                    "SE408", 
                    "SE409", 
                    "SE410", 
                    "SE415", 
                    "SE420", 
                    "SE425", 
                    "SE430", 
                    "SE436", 
                    "SE437", 
                    "SE441", 
                    "SE446", 
                    "SE450", 
                    "SE455", 
                    "SE460", 
                    "SE465", 
                    "SE470", 
                    "SE475", 
                    "SE476", 
                    "SE480", 
                    "SE485", 
                    "SE490", 
                    "SE495", 
                    "SE501", 
                    "SE506", 
                    "SE510", 
                    "SE516", 
                    "SE521", 
                    "SE526", 
                    "SE530", 
                    "SE532", 
                    "SE600", 
                    "SE605", 
                    "SE606", 
                    "SE610", 
                    "SE611", 
                    "SE612", 
                    "SE613", 
                    "SE615", 
                    "SE616", 
                    "SE617", 
                    "SE618", 
                    "SE619", 
                    "SE621", 
                    "SE622", 
                    "SE623", 
                    "SE624", 
                    "SE625", 
                    "SE626", 
                    "SE630", 
                    "SE631", 
                    "SE632", 
                    "SE640", 
                    "SE650", 
                    "SE699", 
                    "SE700", 
                    "SE715", 
                    "SE720", 
                    "SE725", 
                    "SE750", 
                    "SE447", 
                    "SE448", 
                    "SE462", 
                    "SE332", 
                    "SE765", 
                    "SE766", 
                    "SE770", 
                    "SE705", 
                    "SE710", 
                    "SE730", 
                    "SYS02", 
                    "SYS03", ]


NUTS2_conversion = {"(575) Andalucía":              "ES61", 
                    "(515) País Vasco":             "ES21", 
                    "(520) Navarra":                "ES22", 
                    "(505) Asturias":               "ES12", 
                    "(510) Cantabria":              "ES13", 
                    "(525) La Rioja":               "ES23", 
                    "(500) Galicia":                "ES11", 
                    "(545) Castilla y León":        "ES41", 
                    "(550) Madrid":                 "ES30", 
                    "(540) Islas Baleares":         "ES53", 
                    "(535) Cataluña":               "ES51", 
                    "(530) Aragón":                 "ES24", 
                    "(570) Extremadura":            "ES43", 
                    "(560) Comunidad Valenciana":   "ES52", 
                    "(565) Murcia":                 "ES62", 
                    "(555) Castilla-La Mancha":     "ES42", 
                    "(580) Canarias":               "ES70", 
}
