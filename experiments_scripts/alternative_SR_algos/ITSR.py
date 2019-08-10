from optparse import OptionParser
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from itertools import combinations_with_replacement, product
from collections import Counter
import sys

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

class RecursiveSymbolicRegression:

    def __init__( self, LinearModel=Ridge, functions=[np.cos, np.sin, np.tan] ):
        self.model = LinearModel()
        self.functions = functions

    def fit(self, X, y, it_inter=8, it_inv=5, it_trans=3, thr=1e-4):
        self.thr = thr
        expr = self.genInitialExpression(X.shape[1])
        score, lr = self.generateModel(X, y, expr)
        coef = lr.coef_
        candidate_exprs = [expr]
        tabu = {self.printExpr(expr):1}
        bestexpr, bestcoef, bestscore = expr, coef, score

        avail_functions = self.functions
        self.functions = []
        self.use_div = False

        oldcandidates = []

        it = 0
        #print("stage 1")
        while it < it_inter and len(candidate_exprs) and bestscore + sys.float_info.epsilon < 1.0:
            oldcandidates = candidate_exprs.copy()
            candidate_exprs, tabu = self.genCandidates(candidate_exprs, tabu, X, y)
            bestexpr, bestscore, bestcoef = self.updateBestExpr(candidate_exprs, bestexpr, bestscore, bestcoef, X, y)
            it += 1

        if len(candidate_exprs)==0:
            candidate_exprs = oldcandidates.copy()

        self.use_div = True
        it = 0
        #print("stage 2")
        while it < it_inv and len(candidate_exprs) and bestscore + sys.float_info.epsilon < 1.0:
            oldcandidates = candidate_exprs.copy()
            candidate_exprs, tabu = self.genCandidates(candidate_exprs, tabu, X, y)
            bestexpr, bestscore, bestcoef = self.updateBestExpr(candidate_exprs, bestexpr, bestscore, bestcoef, X, y)
            it += 1

        if len(candidate_exprs)==0:
            candidate_exprs = oldcandidates.copy()

        self.functions = avail_functions
        it = 0
        #print("stage 3", len(candidate_exprs))
        while it < it_trans and len(candidate_exprs) and bestscore + sys.float_info.epsilon < 1.0:
            candidate_exprs, tabu = self.genCandidates(candidate_exprs, tabu, X, y)
            bestexpr, bestscore, bestcoef = self.updateBestExpr(candidate_exprs, bestexpr, bestscore, bestcoef, X, y)
            #print(bestscore, bestscore < 1.0)
            it += 1


        _, self.model = self.generateModel(X, y, bestexpr)
        bestexpr = [term for term, coef in zip(bestexpr, self.model.coef_) if
                    np.abs(coef) > self.thr]
        self.fitexpr = bestexpr
        _, self.model = self.generateModel(X, y, bestexpr)
        self.fitcoef = self.model.coef_
        self.fitbias = self.model.intercept_
        self.fitscore = bestscore
        self.terms = len(bestexpr)

    def updateBestExpr(self, candidate_exprs, bestexpr, bestscore, bestcoef, X, y):

        for expr  in candidate_exprs:
            score, lr = self.generateModel(X, y, expr) 
            if score > bestscore:
                bestexpr, bestscore, bestcoef = expr, score, lr.coef_
                #print(bestscore, self.printExpr(expr))
        return bestexpr, bestscore, bestcoef

    def genCandidates(self, candidate_exprs, tabu, X, y): 
        
        candidate_exprs = [newexpr 
                                  for expr in candidate_exprs 
                                  for newexpr in self.expandCandidatesExprs(expr, X, y)
                                  if self.printExpr(newexpr) not in tabu
                              ]
                     
        tabu.update( dict([(self.printExpr(expr), 1) for expr in candidate_exprs]) )

        return candidate_exprs, tabu
        
    def predict(self, X):
        Xt = self.generateNewData(X, self.fitexpr)
        return self.model.predict(Xt)
        
    def passthru(self, X):
        return X

    def printTerm(self, term):
        poly, f = term
        polystr = (' * '.join('x{}**{}'.format(idx,p) for idx, p in poly.items())
                        .replace('**1','')
                  )
        return '{}({})'.format(f.__name__, polystr) if f != self.passthru else polystr

    def printExpr(self, expr, coefs=None, bias=0.0):
        if coefs is None:
            coefs = np.ones(len(expr))
        return ' + '.join(  '{}*{}'.format( round(coef,2), self.printTerm(term) )
                                 for coef, term in zip(coefs, expr) 
                         ) + ' + ' + str(round(bias,2))


    def isSafe(self, X, poly, f):
        """
        Check if it is safe to apply function f to polynomial poly.
        """
        x = self.generateVar(X, (poly, self.passthru))
        y = f(x)
        
        if (y.dtype.char in np.typecodes['AllFloat'] 
                and np.isfinite(y).all() 
                and not np.isnan(y).any()
                and not np.isinf(y).any() 
                and not np.iscomplex(y).any() ):
            return True
        else:
            return False
            

    def generateVar(self, X, term):
        """Generate the new feature of X through a given term."""
        poly, f = term 
        return f( np.prod(
                     [X[:,idx]**p for idx, p in poly.items()], 
                      axis = 0
                  )
                )

    def generateNewData(self, X, expr):
        newdata = [self.generateVar(X, term) for term in expr]
        newdata = np.atleast_2d(newdata).T
        if newdata.shape[0] == 1:
            return newdata.T
        return newdata
        
    def invmae(self, yp, y):
        return 1./ (np.abs(yp-y).mean() + 1.)

    def acc(self, yp, y):
        return (yp==y).sum()

    def generateModel(self, X, y, expr):
        """Generate the Linear Regression Model corresponding to the list of terms."""    
        
        # Generate transformed dataset Xt
        Xt = self.generateNewData(X, expr)
        kf = KFold(n_splits=2)
        #len(Xt)
        calcscore = self.invmae
        score = [ calcscore(self.model
                   .fit(Xt[train_index], y[train_index])
                   .predict(Xt[test_index]), y[test_index])
                   for train_index, test_index in kf.split(Xt)
                ]
        self.model.fit(Xt,y)
        
        # return the worse obtained score and the model
        #score = self.model.score(Xt, y)
        #score = 1. / (np.abs(self.model.predict(Xt) - y).mean() + 1.)
        return np.min(score), self.model
        
    def sumPolys(self, poly1, poly2):
        pcopy = poly1 + poly2    
        pcopy.subtract(-poly1 - poly2)    
        return pcopy.copy()

    def subPolys(self, poly1, poly2):
        pcopy = poly1 - poly2
        pcopy.subtract(poly2 - poly1)
        return pcopy.copy()

    def getInteractions(self, expr):
        """Returns a list of all the interactions among 
        the combination of current terms.
        """
        mult_poly = [ self.sumPolys(poly1,poly2)
                           for (poly1, f1), (poly2, f2) in combinations_with_replacement(expr, 2)
                           if f1==self.passthru and f2==self.passthru
                    ]

        div_poly = [ self.subPolys(poly1,poly2)
                           for (poly1, f1), (poly2, f2) in combinations_with_replacement(expr, 2)
                           if self.use_div and f1==self.passthru and
                           f2==self.passthru
                   ]
        
        interactions = mult_poly + div_poly
        
        return [ (poly, self.passthru) 
                     for poly in interactions 
                     if len(poly)>0 and (poly, self.passthru) not in expr 
               ]
        
    def getTransformations(self, expr, functions):   
        return [ (poly,newf) 
                     for (poly, f), newf in product(expr,functions) 
                     if f!=newf and (poly, newf) not in expr
               ]   
               

    def genInitialExpression(self, nvars):
        """
        Returns the terms x_{i} for i in nvars
        """
        return [ (Counter({idx:1}), self.passthru) for idx in range(nvars) ]

    def getNewScore(self, X, y, expr, term):
        if term is None:
            newscore, _ = self.generateModel(X, y, expr)
        else:
            newscore, _ = self.generateModel(X, y, expr + [term])
        return newscore
        

    def getCandidateTerms(self, expr, heur, score, X):
        all_candidates = self.getInteractions(expr) + self.getTransformations(expr, self.functions)
        candidates_list = [ term for term in all_candidates if self.isSafe(X, *term) ]
        candidates_list = [ (heur(expr, term, score), term) for term in candidates_list ]
        candidates_list = [ term for newscore, term in candidates_list if newscore > min(score,0) ]
        return candidates_list

    def genNewExpr(self, expr, score, candidates_list, heur):
        newexpr = expr.copy()
        unused, newscore = [], score
        
        for term in candidates_list:
            if term not in newexpr:
                gain = max(heur(newexpr, term, newscore), 0.0)
            else:
                gain = 0
            
            if gain > 0:
                newexpr.append(term)
                newscore += gain
            else:
                unused.append(term)

        if len(unused) == len(candidates_list):
            unused = []
        return newexpr, unused

    def expandCandidatesExprs(self, expr, X, y):
        
        score, _ = self.generateModel(X, y, expr)

        def heur(expr, term, score):
            return self.getNewScore(X,y,expr,term) - score

        candidates_list = self.getCandidateTerms(expr, heur, score, X)
        candidate_exprs = [] # [ expr+[term] for term in candidates_list ]
        
        while len(candidates_list) > 0:
            newexpr, candidates_list = self.genNewExpr(expr, score, candidates_list, heur)
            score, lr = self.generateModel(X, y, newexpr)        
            newexpr = [term for term, coef in zip(newexpr, lr.coef_) if
                    np.abs(coef) > self.thr]
            candidate_exprs.append(newexpr)
                
        return candidate_exprs

parser = OptionParser()
parser.add_option("--train", dest="train_fname", help="training data set in CSV format", metavar="FILE")
parser.add_option("--test", dest="test_fname", help="test data set in CSV format", metavar="FILE", default=None)
parser.add_option("--thr", dest="thr", help="term cutoff threshold (defult=1e-4)", default=1e-4, type="float")
parser.add_option("--inter", dest="inter", help="Iterations with just positive interactions (first phase - default=8)", default=8, type="int")
parser.add_option("--inv", dest="inv", help="Iterations with positive and negative interactions (second phase - default=5)", default=5, type="int")
parser.add_option("--trans", dest="trans", help="Iterations with positive and negative interactions and transformation swap (third phase - default=3)", default=3, type="int")
parser.add_option("--reg", dest="reg", help="Linear Regression Model to use: lr: Linear Regression, l1: L1-reg, l2: L2-reg", default="lr")

def pSqRoot(x):
    return np.sqrt(np.abs(x))

def main():
    (options, rags) = parser.parse_args()
    Z_train = np.loadtxt(options.train_fname)
    if options.test_fname is not None:
        Z_test = np.loadtxt(options.test_fname)
    else:
        Z_test = Z_train.copy()

    X_train, y_train = Z_train[:, :-1], Z_train[:, -1]
    X_test, y_test = Z_test[:, :-1], Z_test[:, -1]

    # change this line to use the functions you want
    functions = [np.sin, np.cos, np.tan, pSqRoot, np.log1p, np.log]

    if options.reg == "lr":
        model = LinearRegression
    elif options.reg == "l1":
        model = LassoCV
    elif options.reg == "l2":
        model = RidgeCV
    else:
        model = LinearRegression

    sr = RecursiveSymbolicRegression(LinearModel=model, functions=functions)
    sr.fit(X_train, y_train, options.inter, options.inv, options.trans, options.thr)
    y_pred = sr.predict(X_test)
    mae = np.abs(y_pred-y_test).mean()
    mse = np.square(y_pred-y_test).mean()

    print(f"Test MAE = {mae}, Test MSE = {mse}")

if __name__ == "__main__":
    main()
