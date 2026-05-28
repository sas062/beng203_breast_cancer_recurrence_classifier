import pandas as pd

def main():
    pnas_normal_readcounts = pd.read_csv('data/reads/pnas_normal_tpm.txt', sep='\t')
    pnas_readcounts_96 = pd.read_csv('data/reads/pnas_tpm_96_nodup.txt', sep='\t')

    pnas_normal_readcounts = pnas_normal_readcounts.iloc[1:, :]
    # print(pnas_normal_readcounts.columns.tolist())
    # print(pnas_normal_readcounts.iloc[:5, :3])
    # pnas_normal_readcounts = pnas_normal_readcounts.set_index(pnas_normal_readcounts.columns[0])
    pnas_readcounts_96 = pnas_readcounts_96.set_index(pnas_readcounts_96.columns[0])

    print(pnas_normal_readcounts.shape)
    print(pnas_readcounts_96.shape)


    # pnas_normal_readcounts = pnas_normal_readcounts.loc[:, ~pnas_normal_readcounts.columns.duplicated()]
    # pnas_readcounts_96 = pnas_readcounts_96.loc[:, ~pnas_readcounts_96.columns.duplicated()]

    n_normal = pnas_normal_readcounts.shape[1]
    n_cancer = pnas_readcounts_96.shape[1]

    data = pd.concat([pnas_normal_readcounts, pnas_readcounts_96], axis=1).T

    print(len(data), n_normal + n_cancer)

    # data["cancer_status"] = [0] * n_normal + [1] * n_cancer
    # data["sample_id"] = range(1, len(data) + 1)

    print(data.head())

if __name__ == "__main__":
    main()