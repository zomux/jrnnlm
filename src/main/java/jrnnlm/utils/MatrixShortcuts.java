package jrnnlm.utils;

import org.ejml.data.DenseMatrix64F;

public class MatrixShortcuts {

    public static void addToSpecificCol(DenseMatrix64F matrix, DenseMatrix64F vector, int col, double alpha) {

        // In-place modification
        double[] matrixData = matrix.getData();
        double[] vectorData = vector.getData();
        int rowCount = matrix.numRows;
        int colCount = matrix.numCols;

        for (int row = 0; row < rowCount; ++row) {
            matrixData[col + row * colCount] = alpha * vectorData[row];
        }
    }

    public static void matrixXvector(DenseMatrix64F dest, DenseMatrix64F src, DenseMatrix64F srcMatrix, int from, int to, int from2, int to2) {

        double[] destData = dest.getData();
        double[] srcData = src.getData();
        double[] srcMatrixData = srcMatrix.getData();
        int matrixWidth = srcMatrix.numCols;

        for (int b=from; b<to; b++) {
            for (int a=from2; a<to2; a++) {
                destData[b] += srcData[a] * srcMatrixData[a+b*matrixWidth];
            }
        }
    }

    public static void propagateTwoVectors(DenseMatrix64F matrix, DenseMatrix64F vector1, DenseMatrix64F vector2,  double alpha, int begin1, int end1, int begin2, int end2) {

        double[] matrixData = matrix.getData();
        double[] vector1Data = vector1.getData();
        double[] vector2Data = vector2.getData();
        int colCount = matrix.numCols;

        for (int i = begin1; i < end1; ++i) {
            for (int j = begin2; j < end2; ++j) {

                matrixData[i + j * colCount] += alpha * vector1Data[j] * vector2Data[i];
            }
        }

    }
}
