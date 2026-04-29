// RUN: xdsl-opt %s | filecheck %s

builtin.module {
  %0 = "gem.literal"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>} : () -> tensor<3xf64>
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %{{.+}} = "gem.literal"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>} : () -> tensor<3xf64>
// CHECK-NEXT: }
