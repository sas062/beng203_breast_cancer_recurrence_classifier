import pandas as pd



def main():
    bc_meta = pd.read_csv('data/validation_bc_meta.csv')
    normal_meta = pd.read_csv('data/validation_normal_data.csv')
    pnas_normal_readcounts = pd.read_csv('data/reads/pnas_normal_readcounts.txt', sep='\t')
    pnas_normal_tpm = pd.read_csv('data/reads/pnas_normal_tpm.txt', sep='\t')
    pnas_readcounts_96 = pd.read_csv('data/reads/pnas_readcounts_96.txt', sep='\t')
    pnas_tpm_96 = pd.read_csv('data/reads/pnas_tpm_96.txt', sep='\t')

if __name__ == "__main__":
    main()    
