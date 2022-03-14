use sparse_qr_rust::dtree::DissectionTree;
use sparse_qr_rust::numeric::SparseQR;
use sparse_qr_rust::sparse::CSCSparse;


#[no_mangle]
pub extern "C" fn qrsolve(m : usize, nrhs : usize, nnz : usize, offsets : *const usize, rids : *const usize, vals : *const f64, b : *const f64, x : *mut f64) -> () {
    let offs_slice = unsafe { std::slice::from_raw_parts(offsets,m+1) };
    let rids_slice = unsafe { std::slice::from_raw_parts(rids,nnz) };
    let vals_slice = unsafe { std::slice::from_raw_parts(vals,nnz) };
    let b_slice = unsafe { std::slice::from_raw_parts(b,m*nrhs) };
    let x_slice : &mut [f64] = unsafe { std::slice::from_raw_parts_mut(x,m*nrhs) };

    for (bi,xi) in b_slice.iter().zip(x_slice.iter_mut()){
        *xi = *bi;
    }


    let nrows = m;
    let ncols = m;
    let maxnodes = 200;

    let amat = CSCSparse::<f64>::new(nrows,ncols,offs_slice.to_vec(),rids_slice.to_vec(),vals_slice.to_vec());
    let g = amat.to_metis_graph();
    let g2 = g.square();
    let dtree = DissectionTree::new(&g2,maxnodes);
    let mut fact = SparseQR::<f64>::new(dtree,&amat);

    fact.solve(x_slice);

}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
