from utils.feature_extraction import WordEmbed, Data 

if __name__ == "__main__":
    dat = Data()
    dat.load_data('utils/data.csv')
    arr = dat.tokenize_doc()
    w_e = WordEmbed(arr)
    w_e.get_feature()

