% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sum.R
\name{group_sum}
\alias{group_sum}
\title{Compute a grouped sum}
\usage{
group_sum(x, g)
}
\arguments{
\item{x}{A numeric vector}

\item{g}{A vector defining grouping levels}
}
\description{
Compute a grouped sum
}
\examples{
group_sum(mtcars$cyl, mtcars$vs)
group_sum(mtcars$cyl, mtcars[c("vs", "am")])
}
