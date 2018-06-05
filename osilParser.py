import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import scipy.sparse as sparse

from quadraticConstraint import QuadraticConstraint
from quadraticProblem import QuadraticProblem
from variable import *
from objectiveFunction import *


class OsilParser(object):
    def __init__(self, osilFilePath):
        tree = ET.parse(osilFilePath)
        self.root = tree.getroot()

    def parseProblem(self):
        return self.parseRoot(self.root)

    def parseRoot(self, root):

        instanceData = root[1]

        variableNode = instanceData.find(self.namify('variables'))
        if variableNode is None:
            raise ValueError('The instanceData must contain variables')

        numVars = len(variableNode)
        variables = list(map(self.parseVariable, variableNode))

        # TODO: make helper to avoid duplicate code and reduce line length
        objectives = list(map(lambda obj: self.parseObjective(obj, numVars),
                         instanceData.find(self.namify('objectives'))))

        constraints = list(map(lambda constraint: self.parseConstraint(constraint),
                          instanceData.find(self.namify('constraints'))))

        numConstraints = len(constraints)

        linMatrix = self.parseLinConstraints(
            instanceData.find(self.namify('linearConstraintCoefficients')), numVars, numConstraints)
        quadMatrices = self.parseQuadCoeffs(
            instanceData.find(self.namify('quadraticCoefficients')), numVars)

        # Now need to piece everything together!
        # TODO: Is this going to eat memory for large instances?
        linFuncs = [np.zeros(numVars) for _ in range(numConstraints)]
        print(linFuncs)
        for i, row in enumerate(linMatrix.row):
            linFuncs[row][linMatrix.col[i]] = linMatrix.data[i]

        for i, linFunc in enumerate(linFuncs):
            print(linFunc)
            constraints[i].linear = linFunc

        for index, quadMatrix in quadMatrices.items():
            if index < 0:
                objIndex = index*-1 - 1
                objectives[objIndex].quadratic = quadMatrix
            else:
                constraints[index].quadratic = quadMatrix

        return QuadraticProblem(objectives, constraints, variables)

    @staticmethod
    def parseQuadCoeffs(quadCoeffs, numVars):
        # rows, cols and data representing the sparse matrix
        quadInequalities = defaultdict(lambda: ([], [], []))

        for qTerm in quadCoeffs:
            idxOne = int(qTerm.attrib['idxOne'])
            idxTwo = int(qTerm.attrib['idxTwo'])
            idx = int(qTerm.attrib['idx'])
            coef = int(qTerm.attrib['coef'])
            print((idx, idxOne, idxTwo, coef))

            quadInequalities[idx][0].extend([idxOne, idxTwo])
            quadInequalities[idx][1].extend([idxTwo, idxOne])
            quadInequalities[idx][2].extend([coef, coef])

        print(quadInequalities)
        print([(idx, x) for idx, x in quadInequalities.items()])
        return {idx: sparse.coo_matrix((x[2], (x[0], x[1])), shape=(numVars, numVars))
                for idx, x in quadInequalities.items()}

    @staticmethod
    def namify(s):
        return '{os.optimizationservices.org}' + s

    def parseLinConstraints(self, linConstraints, numVars, numConstraints):
        start = self.elsToList(linConstraints.find(self.namify('start')))
        # Need to insert a leading 0 to match with the numpy sparce matrix representation
        if start[0] != 0:
            start.insert(0, 0)

        values = self.elsToList(linConstraints.find(self.namify('value')))

        colNode = linConstraints.find(self.namify('colIdx'))
        rowNode = linConstraints.find(self.namify('rowIdx'))

        if colNode is not None:
            colIdxs = self.elsToList(colNode)
            return sparse.csr_matrix((values, colIdxs, start), shape=(numConstraints, numVars)).tocoo()
        elif rowNode is not None:
            rowIdxs = self.elsToList(rowNode)
            return sparse.csc_matrix((values, rowIdxs, start), shape=(numConstraints, numVars)).tocoo()
        raise ValueError("The linear constraint must have to have either a colIdx or a rowIdx")

    def elsToList(self, elsNode):
        els = []
        for el in elsNode:
            els.extend(self.elToList(el))
        return els

    @staticmethod
    def elToList(el):
        mult = 1
        incr = 0
        val = int(el.text)
        if 'mult' in el.attrib:
            mult = int(el.attrib['mult'])
        if 'incr' in el.attrib:
            incr = int(el.attrib['incr'])
        return [val + i*incr for i in range(mult)]

    @staticmethod
    def parseVariable(variable):
        print(variable.attrib)
        if 'lb' in variable.attrib:
            lb = variable.attrib['lb']
        else:
            lb = 0

        if 'ub' in variable.attrib:
            ub = variable.attrib['ub']
        else:
            ub = float('inf')

        if 'type' in variable.attrib:
            varType = variable.attrib['type']
        else:
            varType = 'C'

        return Variable(varType, lb, ub)

    def parseObjective(self, objectiveNode, numVars):
        # TODO: Implement this function
        linearTerms = [0]*numVars
        if 'weight' in objectiveNode.attrib:
            weight = objectiveNode.attrib['weight']
        else:
            weight = 1

        if 'name' in objectiveNode.attrib:
            name = objectiveNode.attrib['name']
        else:
            name = 'defaultObjectiveName'

        # Don't need to get this value as we don;t use it
        #if 'numberOfObjCoef' in objectiveNode.attrib:
        #    numberOfObjCoef = objectiveNode.attrib['numberOfObjCoef']
        #else:
        #    numberOfObjCoef = -1

        if 'maxOrMin' in objectiveNode.attrib:
            maxOrMin = objectiveNode.attrib['maxOrMin']
        else:
            maxOrMin = 'min'

        for coeff in objectiveNode:
            linearTerms[int(coeff.attrib['idx'])] = int(coeff.text)

        return ObjectiveFunction(weight, name, maxOrMin, linear=linearTerms)

    @staticmethod
    def parseConstraint(constraintNode):
        if 'name' in constraintNode.attrib:
            name = constraintNode.attrib['name']
        else:
            name = "defaultConstraintName"

        # TODO, deal with the case that we have both lb and ub!
        hasLb = 'lb' in constraintNode.attrib
        hasUb = 'ub' in constraintNode.attrib

        if hasLb and hasUb:
            lb = float(constraintNode.attrib['lb'])
            ub = float(constraintNode.attrib['ub'])
        elif hasUb:
            ub = float(constraintNode.attrib['ub'])
            lb = -float('inf')
        elif hasLb:
            ub = float('inf')
            lb = float(constraintNode.attrib['lb'])
        else:
            raise ValueError('A constraintNode must have at least one of a lower bound or an upper bound')

        return QuadraticConstraint(name, lb, ub)

