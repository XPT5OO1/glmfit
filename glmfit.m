function [bb,dev,stats]=glmfit(x,y,distr,varargin)

  ## Checking number of input arguments
  if (nargin<2)
      error("glmfit: atleast 2 input arguments required.");
  endif

  ## Check supplied parameters
  if ((numel (varargin) / 2) != fix (numel (varargin) / 2))
        error ("glmfit: optional arguments must be in NAME-VALUE pairs.");
  endif
  ## Add defaults
  ## less than 2 arguments then distribution = normal
  if (nargin<3)
      distr= 'normal';
  endif


  link="canonical";
  estdisp="off";
  offset=[];
  weights=[]; ## weights are used when fitting the generalized linear model to give different importance to different observations. If nothing given then all observations are given equal importance when fitting the model.
  constant="on";
  rankwarn="true";
  options=[];
  b0=[];

  params = numel(varargin)
  for idx = 1:2:params
    name = varargin{idx};
    value = varargin{idx+1};
    switch (lower (name))
      case "link"
        link = value;
      case "estdisp"
        estdisp = value;
      case "offset"
        offset = value;
      case "weights"
        weights = value;
      case "constant"
        constant = value;
      case "rankwarn"
        rankwarn = value;
      case {"options"}
        options = value;
      case "b0"
        b0 = value;
      otherwise
        error (sprintf ("glmfit: parameter %s is not supported", name));
    endswitch
  endfor

  ## Adding column of ones if constant is on
  if strcmp(constant, 'on')
      ones_col = ones(size(x, 1), 1);
      x = [ones_col, x]; ## Concatenate the column of ones with x
  end

  ## Separate response variable y and
  ## the number of trials N based on the specified distribution
  N = ones(size(y,1),1);

  if (isfloat(link))
    power_link = link;
    link = "power";
  endif

  ## Check if the link is not canonical
  if (!strcmp(link, "canonical"))
    if (any(strcmp(link, {"identity","log","logit","probit","comploglog","reciprocal","loglog"})))
      ## do nothing continue
    else
      error(" glmfit: Unidentified link");
    endif
      ## Ensure link is a cell array with at least three elements
    if (iscell(link) && numel(link) == 3)
      ## Get the custom link function, its derivative, and its inverse
      link_function = link{1};
      derivative_link = link{2};
      inverse_link = link{3};
    else
      error('glmfit: link: Expected a cell array with three elements.');
    endif
  endif

  distr=tolower(distr);

  if(link=="canonical")
    switch(distr)
      case "normal"
        link = "identity";
        sqrt_var_fun = @(mu) ones(size(mu),"like",mu);
        deviance_function = @(mu,y) (y - mu).^2;
      case "binomial"
        link = "logit";
        sqrt_var_fun = @(mu) sqrt(mu).*sqrt(1-mu) ./ sqrtN;
        deviance_function = @(mu,y) 2*N.*(y.*log(y./mu) + (1-y).*log((1-y)./(1-mu)));
      case "poisson"
        link = "log";
        sqrt_var_fun = @(mu) sqrt(mu);
        deviance_function = @(mu,y) 2* y .* (log(y ./ mu) - 2* (y - mu));
      case "gamma"
        link = "reciprocal";
        sqrt_var_fun = @(mu) mu;
        deviance_function = @(mu,y) 2*((y - mu) ./ mu - log(y ./ mu));
      case "inverse gaussian"
        link = "1/mu^2";
        sqrt_var_fun = @(mu) mu.^(3/2);
        deviance_function = @(mu,y) ((y - mu)./mu).^2 ./  y;
      otherwise
        error ("glmfit: unknown distribution.");
    endswitch
  endif

  switch(link)
      case "identity"
          link_function = @(mu) mu;
          derivative_link = @(mu) ones(size(mu));
          inverse_link = @(eta) eta;
      case "log"
          link_function = @(mu) log(mu);
          derivative_link = @(mu) 1 ./ mu;
          inverse_link = @(eta) exp(eta);
      case "logit"
          link_function = @(mu) log(mu ./ (1-mu));
          derivative_link = @(mu) 1 ./ (mu .* (1-mu));
          inverse_link = @(eta) exp(eta) ./ (1 + exp(eta));
      case "probit"
          link_function = @(mu) norminv(mu);
          derivative_link = @(mu) 1 ./ normpdf(norminv(mu));
          inverse_link = @(eta) normcdf(eta); ##The inverse probit link is the CDF of standard normal distribution.
      case "comploglog"
          link_function = @(mu) log(-log(1-mu));
          derivative_link = @(mu) 1 ./ -((1-mu) .* log(1-mu));
          inverse_link = @(eta) 1 - exp(-exp(eta));
      case "loglog"
          link_function = @(mu) log(-log(mu));
          derivative_link = @(mu)  1 ./ (mu .* log(mu));
          inverse_link = @(eta) exp(-exp(eta ));
      case "reciprocal"
          link_function = @(mu) 1 ./ mu;
          derivative_link = @(mu) -1 ./ mu.^2;
          inverse_link = @(eta) 1 ./ eta;
      case "power"
          link_function = @(mu)  (mu ).^power_link;
          derivative_link = @(mu) power_link * mu.^(power_link-1);
          inverse_link = @(eta)  (eta ) .^ (1/power_link);
      otherwise
          error("glmfit: Unrecognized link");
  endswitch

  if (strcmpi(distr, 'binomial'))
    if (isa(y,'categorical'))
        [y, textname, classname] = grp2idx(y); ##If y is categorical, this line converts y to numeric indices and stores the original category names in textname and classname
        numberofcategories = length(textname);

        if numberofcategories > 2
            error("glmfit: More than two categories.");
        endif
        ## first category mapped to 0 and the second category mapped to 1
        y(y==1) = 0;
        y(y==2) = 1;
    endif
    if size(y,2) == 2
            wasvector = false;
            N = y(:,2);
            y = y(:,1);
    endif
    y = y./N;
  endif


  if(isempty(options))
      iteration_limit =  100;
      convergence_criterion = 1e-6;
  else
      iteration_limit = options.MaxIter;
      convergence_criterion = options.TolX;
  endif


  ## Setting starting value of mu
  if (isempty(b0))
    switch (distr)
      case "poisson"
       mu = y + 0.25;
      case "binomial"
       mu = (N .* y + 0.5) ./ (N + 1);
      case "gamma"
       mu = max(y, eps(y));
      case "inverse gaussian"
       mu = max(y, eps(y));   ##check once again for gamma and inverse gaussian
      otherwise
       mu = y;
    endswitch
      eta = link_Function(mu);
  else
      eta = offset + x * b0(:);
      mu = inverse_link(eta);
  end

  [n_obs,n_pred] = size(x); #n_obs is no. of observations and n_pred is no. of predictor variables
  b = zeros(n_pred,1);

  ## Doing the iterations by Iteratively Reweighted Least Squares (IRLS) algorithm
  for i = 1:iteration_limit

    deta = derivative_link(mu);
    z = eta + (y - mu) .* deta;
    sqrtw = sqrt(weights) ./ (abs(deta) .* sqrt_var_fun(mu));

    ## Performing weighted least squares

    b_old = b;

    [~,p] = size(x);
    yw = (z - offset) .* sqrtw;
    xw = x .* sqrtw(:,ones(1,p));

    if isa(xw, 'gpuArray')
        [Q,R,p] = qr(xw,0);  ## Orthogonal Triangular decomposition
        b(p,:) = R \ (Q'*yw);
        R(p,p) = R;
    else
        [Q,R] = qr(xw,0);  ## Orthogonal Triangular decomposition
        b = R \ (Q'*yw);
    endif

    eta = offset + x * b;
    mu = inverse_link(eta);

    difference = abs(b - b_old);
    threshold = convergence_criterion * max(sqrt(eps), abs(b_old));
    if (all(difference <= threshold))
     break;
    endif
  endfor

endfunction

