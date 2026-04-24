func.func @mykernel(%myvar3:tensor<6xf64>, %myvar5:tensor<3xf64>, %out:tensor<10xf64>) -> tensor<10xf64>
{
%myvar0 = arith.constant dense<[[-0.08347433, 0.07748676,-0.08347433, 0.21266473, 0.06033533, 0.06033533],
 [ 0.33593674,-0.13176445, 0.05032954, 0.43891647,-0.16765117,-0.03041136],
 [ 0.33593674, 0.05032954,-0.13176445,-0.16765117, 0.43891647,-0.03041136],
 [ 0.58045824, 0.58045824, 0.58045824, 0.18495674, 0.18495674, 0.18495674],
 [ 0.05032954,-0.13176445, 0.33593674, 0.43891647,-0.03041136,-0.16765117],
 [-0.13176445, 0.05032954, 0.33593674,-0.16765117,-0.03041136, 0.43891647],
 [-0.08347433,-0.08347433, 0.07748676, 0.06033533, 0.21266473, 0.06033533],
 [ 0.05032954, 0.33593674,-0.13176445,-0.03041136, 0.43891647,-0.16765117],
 [-0.13176445, 0.33593674, 0.05032954,-0.03041136,-0.16765117, 0.43891647],
 [ 0.07748676,-0.08347433,-0.08347433, 0.06033533, 0.06033533, 0.21266473]]> : tensor<10x6xf64>
%myvar1 = arith.constant dense<[0.11169079,0.11169079,0.11169079,0.05497587,0.05497587,0.05497587]> : tensor<6xf64>
%myvar2 = arith.constant -1.0 : f64
%myvar4 = arith.constant dense<[0.44594849,0.10810302,0.44594849,0.81684757,0.09157621,0.09157621]> : tensor<6xf64>
%myvar6 = arith.constant dense<[0.44594849,0.44594849,0.10810302,0.09157621,0.81684757,0.09157621]> : tensor<6xf64>
%myvar7 = arith.constant dense<[0.10810302,0.44594849,0.44594849,0.09157621,0.09157621,0.81684757]> : tensor<6xf64>
%dummy = tensor.empty() : tensor<10x6xf64>
%myresult = linalg.generic
    {
        indexing_maps = [affine_map<(A, B) -> (A, B)>, affine_map<(A, B) -> (A)>],
        iterator_types = ["parallel", "reduction"]
    }
    ins(%dummy: tensor<10x6xf64>)
    outs(%out: tensor<10xf64>)
    {
        ^bb0(%bdummy: f64, %bout : f64) :
        %myvar9 = linalg.index 0 : index
    %myvar10 = linalg.index 1 : index
    %myvar8 = tensor.extract %myvar0[%myvar9, %myvar10] : tensor<10x6xf64>
    %myvar12 = linalg.index 1 : index
    %myvar11 = tensor.extract %myvar1[%myvar12] : tensor<6xf64>
    %myvar14 = arith.constant 0 : index
    %myvar13 = tensor.extract %myvar3[%myvar14] : tensor<6xf64>
    %myvar15 = arith.mulf %myvar2, %myvar13 : f64
    %myvar17 = arith.constant 2 : index
    %myvar16 = tensor.extract %myvar3[%myvar17] : tensor<6xf64>
    %myvar18 = arith.addf %myvar15, %myvar16 : f64
    %myvar20 = arith.constant 1 : index
    %myvar19 = tensor.extract %myvar3[%myvar20] : tensor<6xf64>
    %myvar21 = arith.mulf %myvar2, %myvar19 : f64
    %myvar23 = arith.constant 5 : index
    %myvar22 = tensor.extract %myvar3[%myvar23] : tensor<6xf64>
    %myvar24 = arith.addf %myvar21, %myvar22 : f64
    %myvar25 = arith.mulf %myvar18, %myvar24 : f64
    %myvar27 = arith.constant 0 : index
    %myvar26 = tensor.extract %myvar3[%myvar27] : tensor<6xf64>
    %myvar28 = arith.mulf %myvar2, %myvar26 : f64
    %myvar30 = arith.constant 4 : index
    %myvar29 = tensor.extract %myvar3[%myvar30] : tensor<6xf64>
    %myvar31 = arith.addf %myvar28, %myvar29 : f64
    %myvar33 = arith.constant 1 : index
    %myvar32 = tensor.extract %myvar3[%myvar33] : tensor<6xf64>
    %myvar34 = arith.mulf %myvar2, %myvar32 : f64
    %myvar36 = arith.constant 3 : index
    %myvar35 = tensor.extract %myvar3[%myvar36] : tensor<6xf64>
    %myvar37 = arith.addf %myvar34, %myvar35 : f64
    %myvar38 = arith.mulf %myvar31, %myvar37 : f64
    %myvar39 = arith.mulf %myvar2, %myvar38 : f64
    %myvar40 = arith.addf %myvar25, %myvar39 : f64
    %myvar41 = math.absf %myvar40 : f64
    %myvar42 = arith.mulf %myvar11, %myvar41 : f64
    %myvar44 = linalg.index 1 : index
    %myvar43 = tensor.extract %myvar4[%myvar44] : tensor<6xf64>
    %myvar46 = arith.constant 0 : index
    %myvar45 = tensor.extract %myvar5[%myvar46] : tensor<3xf64>
    %myvar47 = arith.mulf %myvar43, %myvar45 : f64
    %myvar49 = linalg.index 1 : index
    %myvar48 = tensor.extract %myvar6[%myvar49] : tensor<6xf64>
    %myvar51 = arith.constant 1 : index
    %myvar50 = tensor.extract %myvar5[%myvar51] : tensor<3xf64>
    %myvar52 = arith.mulf %myvar48, %myvar50 : f64
    %myvar53 = arith.addf %myvar47, %myvar52 : f64
    %myvar55 = linalg.index 1 : index
    %myvar54 = tensor.extract %myvar7[%myvar55] : tensor<6xf64>
    %myvar57 = arith.constant 2 : index
    %myvar56 = tensor.extract %myvar5[%myvar57] : tensor<3xf64>
    %myvar58 = arith.mulf %myvar54, %myvar56 : f64
    %myvar59 = arith.addf %myvar53, %myvar58 : f64
    %myvar60 = arith.mulf %myvar42, %myvar59 : f64
    %myvar61 = arith.mulf %myvar8, %myvar60 : f64
        %inc = arith.addf %bout, %myvar61: f64
        linalg.yield %inc : f64
    } -> tensor<10xf64>
func.return %myresult : tensor<10xf64>
}