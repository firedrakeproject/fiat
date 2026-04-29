func.func @mykernel(%myvar1:tensor<6xf64>, %out:tensor<3xf64>) -> tensor<3xf64>
{
%myvar0 = arith.constant dense<[1.,0.,0.]> : tensor<3xf64>
%myvar2 = arith.constant dense<[0.,1.,0.]> : tensor<3xf64>
%myvar3 = arith.constant dense<[0.,0.,1.]> : tensor<3xf64>
%dummy = tensor.empty() : tensor<3xf64>
%myresult = linalg.generic
    {
        indexing_maps = [affine_map<(A) -> (A)>, affine_map<(A) -> (A)>],
        iterator_types = ["parallel"]
    }
    ins(%dummy: tensor<3xf64>)
    outs(%out: tensor<3xf64>)
    {
        ^bb0(%bdummy: f64, %bout : f64) :
        %myvar5 = linalg.index 0 : index
    %myvar4 = tensor.extract %myvar0[%myvar5] : tensor<3xf64>
    %myvar7 = arith.constant 0 : index
    %myvar6 = tensor.extract %myvar1[%myvar7] : tensor<6xf64>
    %myvar8 = arith.mulf %myvar4, %myvar6 : f64
    %myvar10 = linalg.index 0 : index
    %myvar9 = tensor.extract %myvar2[%myvar10] : tensor<3xf64>
    %myvar12 = arith.constant 2 : index
    %myvar11 = tensor.extract %myvar1[%myvar12] : tensor<6xf64>
    %myvar13 = arith.mulf %myvar9, %myvar11 : f64
    %myvar14 = arith.addf %myvar8, %myvar13 : f64
    %myvar16 = linalg.index 0 : index
    %myvar15 = tensor.extract %myvar3[%myvar16] : tensor<3xf64>
    %myvar18 = arith.constant 4 : index
    %myvar17 = tensor.extract %myvar1[%myvar18] : tensor<6xf64>
    %myvar19 = arith.mulf %myvar15, %myvar17 : f64
    %myvar20 = arith.addf %myvar14, %myvar19 : f64
        %inc = arith.addf %bout, %myvar20: f64
        linalg.yield %inc : f64
    } -> tensor<3xf64>
func.return %myresult : tensor<3xf64>
}