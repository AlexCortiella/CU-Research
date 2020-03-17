function [q,H,conv] = localSubspace(R,r,epsilon,kh,d)

%Input: 
% R: n x N matrix whose columns are the samples (n is the dimension of the
% ambient space
% r: coordinate where you want to perform the local approximation
% epsilon: tolerance
% d: dimension of the manifold (d < n)
% kh: bandwith of the weighting function

%% Output
% q: an n dimensional vector (origin of the local approximation)
% H: n x d matrix whose columns are the vectors spanning the local subspace
% conv: norm(q-qprev) throughout iterations


%Extract parameters
N = size(R,2);
n = size(R,1);

%Allocate variables
Y = zeros(n,N);

%1 - Spatially Weighted PCA
for i = 1:N
    %Euclidean norm in R^n
    di = norm(R(:,i) - r);
    %Distance-based Weights
    if di > kh
        w = 0;
    else
        w = exp(-di^2/(di-kh)^2);
    end
    %Distance-weighted coordinates
    Y(:,i) = sqrt(w)*R(:,i);
end
%Perform PCA on the distance-weighted coordinates
[Un,S,V] = svd(Y);
U = Un(:,1:d);
% U = rand(n,d);
% U = [-2;1];


%Initial guess for q
q = r;
qprev = ones(n,1);
k = 0;%iteration counter
figure
% plot3(R(1,:),R(2,:),R(3,:),'r.');
[Xm,Ym,Zm] = sphere(20);
surf(Xm,Ym,Zm,'EdgeColor','none','FaceAlpha',0.5)
hold on
plot3(q(1),q(2),q(3),'go');
quiver3(q(1),q(2),q(3),Un(1,1),Un(2,1),Un(3,1));
quiver3(q(1),q(2),q(3),Un(1,2),Un(2,2),Un(3,2));
quiver3(q(1),q(2),q(3),Un(1,3),Un(2,3),Un(3,3));
axis('equal')
pause
while norm(q - qprev) >= epsilon
    qprev = q;
    Rtilde = R - repmat(q,1,N); %Shift origin (Shift data relative to the origin at iteration k)
    %Form weight matrix
    dist = sqrt(sum(Rtilde.*Rtilde,1));%Euclidean distances relative to the origin at iteration k
    theta = exp(-dist.^2./(dist-kh).^2);
    ind = find(dist > kh);
    theta(ind) = 0;%Finite support (indicator function)
    Theta = diag(sqrt(theta));
    %Weight the sample matrix by Theta
    Rtilde = Rtilde*Theta;%Weighted relative coordinates
    %Find the representation of ri-qj (relative coordinates) in Col(U);
    X = Rtilde'*U;%Projected coordinates of ri onto U relative to qj
    %Define Xtilde (linear least squares matrix)
    Xtilde = [ones(N,1),X];% constant and linear terms
    %Solve the LS minimization of Xtilde*alpha = Rtilde'
    alpha = Xtilde\Rtilde';
    %Update the origin of the local coordinate system
    qtilde = q + alpha(1,:)';%First row of alpha corresponding to the constant term (absolute origin previous origin + relative(updated) origin)
    %Form basis matrix
    B = alpha(2:end,:)' - repmat(qtilde,1,d);
    %QR decomposition of B
    [Q,Rhat] = qr(B);
    %Update local basis
    U = Q(:,1:d);
    %Enforce orthogonality r-q orth to H
    q = qtilde + U*U'*(r-qtilde);
    k = k + 1;
    conv(k) = norm(q - qprev);
    fprintf(['Iteration #: ',num2str(k),'\n'])
    fprintf(['Convergence: ',num2str(conv(k)),'\n\n'])
%     plot3(R(1,:),R(2,:),R(3,:),'r.');
    plot3(q(1),q(2),q(3),'go');
    quiver3(q(1),q(2),q(3),Q(1,1),Q(2,1),Q(3,1));
    quiver3(q(1),q(2),q(3),Q(1,2),Q(2,2),Q(3,2));
    quiver3(q(1),q(2),q(3),Q(1,3),Q(2,3),Q(3,3));
    drawnow
    axis('equal')
%     plot(R(1,:),R(2,:),'r.');
%     hold on
%     plot(q(1),q(2),'go');
%     quiver(q(1),q(2),Q(1,1),Q(2,1));
%     axis('equal')
%     pause
end

H = U;

end

