% Evaluates a multi_D monomial basis at a given point (xi_1,...,xi_d)

function pc_xi = piset_monomial(xi,index_pc)

pc_xi = ones(size(index_pc,1),1);

p = sum(index_pc(size(index_pc,1),:));
Monomial = my_monomial_1d(p,xi);

for id=1:size(index_pc,2);
    nnz_index = find(index_pc(:,id)>0);
    if find(index_pc(:,id)>0)
        pc_xi(nnz_index) = pc_xi(nnz_index).*Monomial(index_pc(nnz_index,id)+1,id);
    end
end
