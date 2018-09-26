#include "discrete_operators.h"

namespace GC {

    //using namespace Eigen;

    // Hodge star on 0-forms. Returns a (nVerts, nVerts) matrix.
    template<typename T, typename D>
    GC::SparseMatrix<T> buildHodge0(Geometry<D>* geometry) {

        HalfedgeMesh* mesh = geometry->getMesh();
        VertexData<size_t> vInd = mesh->getVertexIndices();
        size_t nVerts = mesh->nVertices();

        // Reserve space in the sparse matrix
        GC::SparseMatrix<T> hodge0 = GC::SparseMatrix<T>(nVerts, nVerts);

        for(VertexPtr v : mesh->vertices()) {

            double primalArea = 1.0;
            double dualArea = geometry->area(v.dual());
            double ratio = dualArea / primalArea;
            size_t iV = vInd[v];
            hodge0(iV, iV) = ratio;

        }

        return hodge0;
    }
    template GC::SparseMatrix<double> buildHodge0<double, Euclidean>(Geometry<Euclidean>* geometry);
    template GC::SparseMatrix<Complex> buildHodge0<Complex, Euclidean>(Geometry<Euclidean>* geometry);


    // Hodge star on 1-forms. Returns a (nEdges, nEdges) matrix.
    template<typename T, typename D>
    GC::SparseMatrix<T> buildHodge1(Geometry<D>* geometry) {
        
        HalfedgeMesh* mesh = geometry->getMesh();
        EdgeData<size_t> eInd = mesh->getEdgeIndices();
        size_t nEdges = mesh->nEdges();

        GC::SparseMatrix<T> hodge1 = GC::SparseMatrix<T>(nEdges, nEdges);

        // Get the cotan weights all at once
        EdgeData<double> cotanWeights(mesh);
        geometry->getEdgeCotanWeights(cotanWeights);

        for(EdgePtr e : mesh->edges()) {

            double ratio = cotanWeights[e];
            size_t iE = eInd[e];
            hodge1(iE, iE) = ratio;
        }


        return hodge1;
    }
    template GC::SparseMatrix<double> buildHodge1<double, Euclidean>(Geometry<Euclidean>* geometry);
    template GC::SparseMatrix<Complex> buildHodge1<Complex, Euclidean>(Geometry<Euclidean>* geometry);


    // Hodge star on 2-forms. Returns a (nFaces, nFaces) matrix.
    template<typename T, typename D>
    GC::SparseMatrix<T> buildHodge2(Geometry<D>* geometry) {

        HalfedgeMesh* mesh = geometry->getMesh();
        FaceData<size_t> fInd = mesh->getFaceIndices();
        size_t nFaces = mesh->nFaces();

        // Reserve space in the sparse matrix
        GC::SparseMatrix<T> hodge2 = GC::SparseMatrix<T>(nFaces, nFaces);

        for(FacePtr f : mesh->faces()) {

            double primalArea = geometry->area(f);
            double dualArea = 1.0;
            double ratio = dualArea / primalArea;

            size_t iF = fInd[f];
            hodge2(iF, iF) = ratio;
        }
    

        return hodge2;

    }
    template GC::SparseMatrix<double> buildHodge2<double, Euclidean>(Geometry<Euclidean>* geometry);
    template GC::SparseMatrix<Complex> buildHodge2<Complex, Euclidean>(Geometry<Euclidean>* geometry);


    // Derivative on 0-forms. Returns a (nEdges, nVerts) matrix
    template<typename T>
    GC::SparseMatrix<T> buildDerivative0(HalfedgeMesh* mesh) {

        VertexData<size_t> vInd = mesh->getVertexIndices();
        EdgeData<size_t> eInd = mesh->getEdgeIndices();
        size_t nVerts = mesh->nVertices();
        size_t nEdges = mesh->nEdges();

        GC::SparseMatrix<T> d0 = GC::SparseMatrix<T>(nEdges, nVerts);

        for(EdgePtr e : mesh->edges()) {

            size_t iEdge = eInd[e];
            HalfedgePtr he = e.halfedge();
            VertexPtr vTail = he.vertex();
            VertexPtr vHead = he.twin().vertex();

            size_t iVHead = vInd[vHead];
            d0(iEdge, iVHead) = 1.0;

            size_t iVTail = vInd[vTail];
            d0(iEdge, iVTail) = -1.0;

        }

        return d0;
    }
    template GC::SparseMatrix<double> buildDerivative0<double>(HalfedgeMesh* mesh);
    template GC::SparseMatrix<Complex> buildDerivative0<Complex>(HalfedgeMesh* mesh);


    // Derivative on 1-forms. Returns a (nFaces, nEdges) matrix
    template<typename T>
    GC::SparseMatrix<T> buildDerivative1(HalfedgeMesh* mesh) {

        EdgeData<size_t> eInd = mesh->getEdgeIndices();
        FaceData<size_t> fInd = mesh->getFaceIndices();
        size_t nEdges = mesh->nEdges();
        size_t nFaces = mesh->nFaces();

        GC::SparseMatrix<T> d1 = GC::SparseMatrix<T>(nFaces, nEdges);

        for(FacePtr f : mesh->faces()) {

            size_t iFace = fInd[f];

            for(HalfedgePtr he : f.adjacentHalfedges()) {

                size_t iEdge = eInd[he.edge()];
                double sign = (he == he.edge().halfedge()) ? (1.0) : (-1.0);
                d1(iFace, iEdge) = sign;

            }

        }

        return d1;
    }
    template GC::SparseMatrix<double> buildDerivative1<double>(HalfedgeMesh* mesh);
    template GC::SparseMatrix<Complex> buildDerivative1<Complex>(HalfedgeMesh* mesh);

}