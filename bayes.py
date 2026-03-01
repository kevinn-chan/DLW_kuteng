def bayes(pT, pVgivenT, pV):
    if pV == 0:
        raise ValueError("P(V) cannot be zero")
    pTgivenV = (pVgivenT * pT) / pV
    return pTgivenV

def compute_pV(pT, pVgivenT, pVgiven_notT):
    p_notT = 1 - pT
    return (pVgivenT * pT) + (pVgiven_notT * p_notT)

pT = 0.01
pVgivenT = 0.99
pV = 0.0198

result = bayes(pT, pVgivenT, pV)
print(result)

'''
pV: prob of visual evidence occuring anytime
pT: mean expected threat prob (from monte carlo)
pVgivenT: prob of visual evidence occuring anytime given a threat (raw anomaly score)
pTgivenV: prob of threat given visual evidence
'''