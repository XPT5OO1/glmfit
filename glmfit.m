##if(0)
## Copyright (C) 2024 Andreas Bertsatos <abertsatos@biol.uoa.gr>
## Copyright (C) 2024 Pallav Purbia <pallavpurbia@gmail.com>
##
## This file is part of the statistics package for GNU Octave.
##
## Octave is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## Octave is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not,
## see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {statistics} {@var{b} =} glmfit (@var{X}, @var{y}, @var{distribution})
## @deftypefnx {statistics} {@var{b} =} glmfit (@var{X}, @var{y}, @var{distribution},@var{Name}, @var{Value})
## @deftypefnx {statistics} {[@var{b}, @var{dev}] =} glmfit (@dots{})
##
## Perform generalized linear model fitting.
##
## @code{@var{b} = glmfit (@var{X}, @var{y}, @var{distribution})} returns a
## coefficient estimates vector, @var{b} for a
## generalized linear regression model of responses in @var{y} and
## predictors in @var{X}, using the @var{distribution}.
##
## @itemize
## @item @var{X} is an @math{nxp} numeric matrix of predictor variables with
## @math{n} observations and @math{p} predictors.
## @item @var{y} is a @math{n X 1} numeric vector of responses for all supported
## distributions, except for the 'binomial' distribution which can also have
## @var{y} as a @math{n X 2} matrix, where the first column contains the number
## of successes and the second column contains the number of trials.
## @item @var{distribution} specifies the distribution of the response variable
## (e.g., 'poisson').
## @end itemize
##
## @code{@var{b} = glmfit (@dots{}, @var{Name}, @var{Value})}
## specifies additional options using @qcode{Name-Value} pair arguments.
##
## @multitable @columnfractions 0.18 0.02 0.8
## @headitem @var{Name} @tab @tab @var{Value}
##
## @item @qcode{"link"} @tab @tab Specifies the relationship f(mu)=X*b. It can be:
## @itemize
## @item A character vector specifying a link
## function from the list of inbuilt links 'identity', 'log', 'logit',
## 'probit', 'comploglog', 'reciprocal', 'loglog' or,
## @item Custom link provided as a structure with three fields:
## Link Function, Derivative Function, Inverse Function. or,
## @item Exponent P, such that the relationship is @math{mu = (X*b)^P}
## @end itemize
##
## @item @qcode{"estdisp"} @tab @tab Specifies whether to
## estimate the dispersion parameter for the Binomial
## or Poisson distribution. Options are
## @var{"on"} or @var{"off"} (default). If @var{"on"},
## the dispersion parameter will be estimated from the data.
## If @var{"off"}, the dispersion parameter will not be estimated.
##
## @item @qcode{"offset"} @tab @tab An vector to be added to a linear predictor
## with known coefficient 1 rather than an estimated coefficient.
##
## @item @qcode{"weights"} @tab @tab An optional vector of prior weights
## to be k in the fitting process.
##
## @item @qcode{"constant"} @tab @tab Specifies whether to
## include a constant term in the model. Options are
## @var{"on"} (default) or @var{"off"}.
##
## @item @qcode{"options"} @tab @tab A structure specifying
## control parameters for the iterative algorithm with the
## following field and its default value:
## @itemize
## @item @qcode{@var{options}.MaxIter = 100}
## @item @qcode{@var{options}.TolX = 1e-6}
## @end itemize
##
## @item @qcode{"b0"} @tab @tab Starting values for the
## estimated coefficients for the iteration.
## @end multitable
##
## @code{[@var{b}, @var{dev}] = glmfit (@dots{})}
## returns the estimated coefficient vector, @var{b}, as well as
## the deviance, @var{dev}, of the fit.
##
## Supported distributions include 'normal, 'binomial', 'poisson', 'gamma', and 'inverse gaussian'.
## Supported link functions include 'identity', 'log', 'logit', 'probit',
## 'loglog', 'comploglog', 'reciprocal', custom link and Power link.
## Custom link function provided as a structure with three fields:
## Link Function, Derivative Function, Inverse Function.
## @end deftypefn
##endif

function [bb,dev,stats]=glmfit(X,y,distr,varargin)

## Check input
  if (nargin < 2)
    error ("glmfit: too few input arguments.");
  elseif (mod (nargin - 3, 2) != 0)
    error ("glmfit: optional arguments must be in NAME-VALUE pairs.");
  elseif (! isnumeric (X))
    error ("glmfit: X must be a numeric.");
  elseif (! isnumeric (y))
    error ("glmfit: Y must be a numeric.");
  elseif (size (X, 1) != size (y, 1))
    error ("glmfit: X and Y must have the same number of observations.");
  elseif (! ischar (distr))
    error ("glmfit: DISTRIBUTION must be a character vector.");
  endif

## Adding Defaults
  if (nargin<3)
      distr= "normal";
  endif

  link="canonical";
  estdisp="off";
  offset=[];
  weights=[]; ## weights are k when fitting the generalized linear model to give different importance to different observations. If nothing given then all observations are given equal importance when fitting the model.
  constant="on";
  options=[];
  b0=[];

  params = numel(varargin);
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
      case "options"
        if(! isstruct (options) || ! isfield (options, "MaxIter") || ! isfield (options, "TolX"))
        error (strcat (["glmfit: 'options' must be a"], ...
                     [" structure with "], ...
                     [" 'MaxIter', and 'TolX' fields present."]));
        endif
        options = value;

      case "b0"
        b0 = value;
      otherwise
        error (sprintf ("glmfit: parameter '%s' is not supported.", name));
    endswitch
  endfor

  if (isempty(options))
    options.MaxIter = 100;
    options.TolX = 1e-6;
  endif

  ## Adding column of ones if constant is on
  if strcmp(constant, 'on')
      ones_col = ones(size(X, 1), 1);
      X = [ones_col, X]; ## Concatenate the column of ones with X
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
    elseif(iscell(link))
      if (numel(link) == 3)
        ## Get the custom link function, its derivative, and its inverse
        link_function = link{1};
        derivative_link = link{2};
        inverse_link = link{3};
      else
        error('glmfit: link: Expected a cell array with three elements.');
      endif
    else
      error("glmfit: Unidentified link.");
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

  iteration_limit = options.MaxIter;
  convergence_criterion = options.TolX;



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
      eta = offset + X * b0(:);
      mu = inverse_link(eta);
  end

  [n_obs,n_pred] = size(X); #n_obs is no. of observations and n_pred is no. of predictor variables
  b = zeros(n_pred,1);

  ## Doing the iterations by Iteratively Reweighted Least Squares (IRLS) algorithm
  for i = 1:iteration_limit

    deta = derivative_link(mu);
    z = eta + (y - mu) .* deta;
    sqrtw = sqrt(weights) ./ (abs(deta) .* sqrt_var_fun(mu));

    ## Performing weighted least squares

    b_old = b;

    [~,p] = size(X);
    yw = (z - offset) .* sqrtw;
    xw = X .* sqrtw(:,ones(1,p));

    if isa(xw, 'gpuArray')
        [Q,R,p] = qr(xw,0);  ## Orthogonal Triangular decomposition
        b(p,:) = R \ (Q'*yw);
        R(p,p) = R;
    else
        [Q,R] = qr(xw,0);  ## Orthogonal Triangular decomposition
        b = R \ (Q'*yw);
    endif

    eta = offset + X * b;
    mu = inverse_link(eta);

    difference = abs(b - b_old);
    threshold = convergence_criterion * max(sqrt(eps), abs(b_old));
    if (all(difference <= threshold))
     break;
    endif
  endfor

   STATS = struct ("dfe", ...
                  "s", [], ...
                  "sfit", [], ...
                  "se", [], ...
                  "coeffcorr", [], ...
                  "covb", [], ...
                  "t", [], ...
                  "p", [], ...
                  "resid", [], ...
                  "residp", [], ...
                  "residd", [], ...
                  "resida", [] ...
                  )
##       'dfe'       degrees of freedom for error
##       's'         theoretical or estimated dispersion parameter
##       'sfit'      estimated dispersion parameter
##       'se'        standard errors of coefficient estimates B
##       'coeffcorr' correlation matrix for B
##       'covb'      estimated covariance matrix for B
##       't'         t statistics for B
##       'p'         p-values for B
##       'resid'     residuals
##       'residp'    Pearson residuals
##       'residd'    deviance residuals
##       'resida'    Anscombe residuals
endfunction

##if(0)
## Test input validation
%!error <glmfit: too few input arguments.> glmfit ()
%!error <glmfit: too few input arguments.> glmfit (1)
%!error <glmfit: optional arguments must be in NAME-VALUE pairs.> ...
%! glmfit (rand (6, 1), rand (6, 1), 'poisson', 'link')
%!error <glmfit: X must be a numeric.> ...
%! glmfit ('abc', rand (6, 1), 'poisson')
%!error <glmfit: Y must be a numeric.> ...
%! glmfit (rand (5, 2), 'abc', 'poisson')
%!error <glmfit: X and Y must have the same number of observations.> ...
%! glmfit (rand (5, 2), rand (6, 1), 'poisson')
%!error <glmfit: DISTRIBUTION must be a character vector.> ...
%! glmfit (rand (6, 2), rand (6, 1), 3)
%!error <glmfit: DISTRIBUTION must be a character vector.> ...
%! glmfit (rand (6, 2), rand (6, 1), {'poisson'})
%!error <glmfit: for a binomial distribution, Y must be an n-by-1 or n-by-2 matrix.> ...
%! glmfit (rand (5, 2), rand (5, 3), 'binomial')
%!error <glmfit: for distributions other than the binomial, Y must be an n-by-1 column vector> ...
%! glmfit (rand (5, 2), rand (5, 2), 'normal')
%!error <glmfit: 'gamma' distribution is not supported yet.> ...
%! glmfit (rand (5, 2), rand (5, 1), 'gamma')
%!error <glmfit: 'inverse gaussian' distribution is not supported yet.> ...
%! glmfit (rand (5, 2), rand (5, 1), 'inverse gaussian')
%!error <glmfit: unknown distribution.> ...
%! glmfit (rand (5, 2), rand (5, 1), 'loguniform')
%!error <glmfit: link: Expected a cell array with three elements.>...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', {'log'})
%!error <glmfit: custom link functions must be in a three-element cell array.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', {'log', 'hijy'})
%!error <glmfit: custom link functions must be function handles.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', {'log','dfv','dfgvd'})
%!error <glmfit: custom link functions must be function handles.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', {@log, 'derivative', @exp})
%!error <glmfit: custom inverse link function must return output of the same size as input.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', {@exp, @log, @(X) eye(e)})
%!error <glmfit: Unidentified link.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', 'somelinkfunction')
%!error <glmfit: invalid value for link function.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', 2)
%!error <glmfit: constant should be either 'on' or 'off'.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', 'log', 'constant', 0)
%!error <glmfit: constant should be either 'on' or 'off'.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'link', 'log', 'constant', 'asda')
%!error <glmfit: unknown parameter name.> ...
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'param', 'log', 'constant', 'on')
%!error <glmfit: parameter 'someparameter' is not supported.>
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'someparameter', 2)
%!error <glmfit: 'options' must be a structure with 'MaxIter', and 'TolX' fields present.>
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'options', s=struct('MaxIter', [], 't', []))
%!error <glmfit: 'options' must be a structure with 'MaxIter', and 'TolX' fields present.>
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'options', s=struct('M', [], 'TolX', []))
%!error <glmfit: 'options' must be a structure with 'MaxIter', and 'TolX' fields present.>
%! glmfit (rand(5,2), rand(5,1), 'poisson', 'options', 'not a structure')
##endif
