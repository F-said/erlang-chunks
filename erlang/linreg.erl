-module(linreg).
-export([
	start/2,
	learn/6,
	calc_ind/3,
	worker_ind/3,
	calc_batch/4,
	%calc_sect/3,
	collect/2
]).

-record(dataset, {x=[], y=[]}).
-record(model, {b0, b1}).
	
start(Input, Output) ->
	% read CSV file in and save into `dataset` record for easy ref
	{_, CSV_Bin} = file:read_file(Input),
	Data = binary_to_list(CSV_Bin),
	
	Rows = [ Row || Row <- string:split(Data, "\n", all), Row /= ""],
	
	% accumulate each positional value into an accumulate for X & Y
	Dataset = #dataset{
		x = lists:foldl(fun(X, Acc) -> Acc ++ [list_to_float(lists:nth(1, string:split(X, ",", all)))] end, [], Rows),		
		y = lists:foldl(fun(X, Acc) -> Acc ++ [list_to_float(lists:nth(2, string:split(X, ",", all)))] end, [], Rows)
	}, 
	
	% prep Parent PID for signaling
	Host = self(),
	
	% make empty model to prep for prediction
	Model = #model{b0=0, b1=0},
	
	% "learn" optimal regression model, on my dataset, learning rate of 0.1, 20 iterations
	% true for verbosity
	Trained = learn(Dataset, 0.01, Model, Host, 10, true),
	% print out how many processes printed out
	io:format("Processes: ~w~n", [length(erlang:processes())]),
	
	% predict values for trained model
	Predict = calc_ind(
		Dataset#dataset.x, 
		fun(X) -> Trained#model.b1 * X + Trained#model.b0 end, 
		Host
	),

	% write out
	File1 = [io_lib:format("~w~n", [P]) || P <- Predict],
	file:write_file(Output, File1).
	

learn(_, _, Model, _, 0, _) -> Model; 
learn(Data, L, Model, Host, Iter, Verbose) ->
	% a learning algorithm for linear regression that makes individual processes
		% for each data-point
	% Data: Dataset of X & Y 
	% L: Learning rate
	% Model: Model to assign weights to
	% Host: Main process
	% Iter: For lack of a good algo to stop learning, iterations
	% Verbose: Print info as iterates?
	% Worker: Which worker process to use (individual/batch)?
	
	% predict values using current model	
	% get a list of predicted values
	Predicted = calc_ind(
		Data#dataset.x, 
		fun(X) -> Model#model.b1 * X + Model#model.b0 end, 
		Host
	),
	
	% calc partial derivative of bias
	CompareC = [[Y, P] || P <- Predicted, Y <- Data#dataset.y],
	PartialC = calc_batch(
		CompareC,
		fun([X, Y]) -> X - Y end,
		Host,
		10
	), 
	Dc = (-2 * lists:sum(PartialC))/length(PartialC),
	Newb0 = Model#model.b0 - (L * Dc),
	
	% calc partial derivative of coeff
	Compare = [[X, Y, P] || P <- Predicted, Y <- Data#dataset.y, X <- Data#dataset.x],
	PartialM = calc_ind(
		Compare,
		fun([X, Y, Z]) -> X * (Y - Z) end,
		Host
	), 
	Dm = (-2 * lists:sum(PartialM))/length(PartialM),
	Newb1 = Model#model.b1 - (L * Dm),
	
	% adjust weights
	NewModel = #model{b0=Newb0, b1=Newb1},
	
	% print if verbose
	if
	Verbose -> 
		io:format("Iter ~w: B0: ~w B1: ~w~n", [Iter, NewModel#model.b0, NewModel#model.b1]);
	% returning an atom if verbosity isnt selected (ridiculous)
	true -> 
		none
	end,
	
	% learn again
	learn(Data, L, NewModel, Host, Iter-1, Verbose).
	
calc_ind(X, F, Host) ->
	% a function that creates as many processes as values
	% returns a new list with calculated values
	Pids = [spawn(fun() -> worker_ind(Host, F, X1) end) || X1 <- X ],

	collect(Pids, []).

worker_ind(Host, Fun, D) ->
	Res = Fun(D),
	Host ! { single, Res }.
	
calc_batch(X, F, Host, Size) ->
	% a function that interprets partitions of the dataset as batches
	Partitions = [ lists:sublist(X, Ind, Size) || Ind <- lists:seq(1,length(X),Size) ],
	Partitions.

worker_batch(Host, Fun, D) ->
	Res = Fun(D),
	Host ! { single, Res }.

collect([], Vals) -> Vals;
collect([ _ | Tail ], Vals) ->
    receive 
    	{single, Res} ->
    		Newlist = lists:append(Vals, [Res]),
            	collect(Tail, Newlist)
    end.
