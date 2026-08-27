We shouldn't assume that det(J) > 0.  The UFC ordering we are explicitly utilizing means that we will have both signs.  Even in 2d, some triangles will be ordered counterclockwise and others clockwise.  This appears in a few places in the document.  Note that FIAT's reference normals may be inward or outward pointing (based on UFC convention) and that the existing zany maps in FInAT work without this assumption.

We need to define "facet frame" explicitly.



Every functional in (2.8) has exactly one direction and order of differentiation.  This is probably sufficient for all the nodes we're interested in, but we may find a situation where this needs to be generalized.  Don't generalize yet just to do so.

After (2.9), we say "dividing by J".  Change that to multiplying by J^{-T}.

Be more clear in this paragraph -- push-forward changes the direction and so the actual transformation is geometrically dependent, even if we are doing a lot of preprocessing that can be done numerically.

Be more clear in the "The design equation" -- we say "conversely", but what are we contrasting with.

Be more clear about (2.10) -- it's "solvable", but are we going to do that numerically for coefficients of the directions or do we have to do symbolic processing (this will be clear later, but we need to cue the reader as to what's coming)

Is the direction tensor d always going to be symmetric, and if it's outer(n, n) (say) in 2d, then it has rank 1 instead of 2.  Is this reduced rank something that generalizes to other situations (e.g. Wu-Xu) and does it give us anything important to consider?

Above (2.11) -- "scaled tangent" is FIAT-internal lingo (unit vector times length/measure).  Worth defining explicitly.

Above (2.11) point #3 -- I presume this includes integral moments of derivatives as well?

Another point in this discussion -- if we're doing things numerically, without dispatching on type, how do we know that we have a normal component.  I think the point is that we don't know and don't need to.  Instead, we're giving suitable bases in reference and physical configurations in which to expand whatever direction we're working with.

Notation: using big N for the physical normal and little \hat{n} for reference seems odd. Of course, we count things n.  Maybe using \nu and \hat{\nu} for physical/reference normals so we don't risk confusion between N and the set of nodes \mathcal{N}?


For facet nodes and completion, does this work for derivative nodes that are linear combinations of point derivatives (e.g. integral moments)?
In that case, do we reproduce the fundamental theorem of calculus, writing the completed derivative as the difference between point values on the edges?
This is claimed after (2.21), but a proof that we get FTOC would be nice.


Is there a nice analog of some Stokes-like theorem for an integral moment of a normal derivative on a face of a tetrahedron?  I think there is something like this in Xu's papers on generalized nonconforming elements.  Again, it would be nice to show (a theorem?) that we get that from our numerics in the special case.

Is it necessary at all to "bin" nodes that have the same set of points and/or weights, or does this sort of fall out numerically?



I also think the recursive reduction to form the V matrix product without E V^c D being all explicit is very nice.  However, I think it's worth separating the
presentation into two parts:
1) exactly describing how we can form the three bits E V^c and D separately -- V^c essentially follows from the affine-interpolation equivalent discussion, and D is described by the numerical systems we solve, for example.  This could be more explicit.
2) then keep describing the reduction via row recursion as a separate step.  It's got to be morally equivalent to some graph algorithm for a sparse triple product, but with blocks based on cell topology somehow. 

(2.23) again may have to deal with the possibility that these matrices have negative determinant?

Phrase the recursion in pseudocode (algorithmic in latex)

It would be nice to work out (explicitly with formulas?) the Morley and maybe hexic Argryis to demonstrate the process.
Then, we should put in an example of an element whose transformation is *not* explicitly known in the literature (e.g. one of Xu's generalized nonconforming elements?)


Stylistic point: is it worth identifying propositions/theorems about the calcultaions we do to, say, identify the structure/entries of V^c, E, D, and the recursive algorithm for the product?

