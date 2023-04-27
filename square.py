import numpy as np
from scipy.stats import chi2_contingency

def chi_sq_test(modelA_acc = float,modelB_acc =float,n = int):
    '''
    這段程式碼是實現卡方檢定(chi-squared test)的功能，用於比較兩個二元事件之間的關係是否有統計學意義。
    
    在這個函數中，傳入三個參數：
        modelA_acc: 事件A發生的機率
        modelB_acc: 事件B發生的機率
        n: 事件發生的總次數
    函數首先使用 numpy 庫中的 chi2_contingency() 函數來計算兩個事件之間的卡方值和 p 值，
    然後根據 p 值的大小來判斷兩個事件是否有統計學意義。
    如果 p 值小於 0.05，    則認為結果是有統計學意義的，否則結果是無統計學意義的。

    最後，函數返回一個布爾值和 p 值，表示兩個事件之間是否有統計學意義以及相應的 p 值是多少。在這個例子中，
    傳入的參數是 modelA_acc = 0.84、modelB_acc = 0.88、n = 645，函數執行後輸出 p_val 的值以及是否有統計學意義。
    '''
    _, p_val, _, _ = chi2_contingency(np.array([[n*modelA_acc, n*(1-modelA_acc)],
                                                [n*modelB_acc, n*(1-modelB_acc)]]))
    if p_val < 0.05:
        print('p_val : ',p_val,"\n顯著")  
    else:
        print('p_val : ',p_val,"\n不顯著")
    return True if p_val < 0.05 else False,p_val

chi_sq_test(0.84,0.88,645)
