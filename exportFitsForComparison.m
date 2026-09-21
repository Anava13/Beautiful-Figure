function exportFitsForComparison(fitResult, csvPath)
% exportFitsForComparison  Write one row per entry component from the MATLAB
%   fitPlateMixtures output, for comparison with the Python fits.
%
%   fits = fitPlateMixtures(readFigureWorkbook('Input.xlsx'));
%   exportFitsForComparison(fits, 'matlab_fits.csv');
%
%   Columns: entry_id, k, component (1 = smallest median), median, weight,
%   p25, p75.

    % === COORDINATE SUB-CONTROLLERS ===
    comparisonTable = buildComparisonTable(fitResult);
    writeComparisonTable(comparisonTable, csvPath);

    % === SUB-CONTROLLERS ===

    function T = buildComparisonTable(fitResult)
        % MODEL (TERMINAL): one row per component of every plate.
        plates = fitResult.plates;
        nRows  = sum([plates.nComponents]);
        entryID   = cell(nRows, 1);
        kCol      = zeros(nRows, 1);
        component = zeros(nRows, 1);
        medianCol = zeros(nRows, 1);
        weightCol = zeros(nRows, 1);
        p25Col    = zeros(nRows, 1);
        p75Col    = zeros(nRows, 1);
        r = 0;
        for p = 1:numel(plates)
            for c = 1:plates(p).nComponents
                r = r + 1;
                entryID{r}   = plates(p).plateID;
                kCol(r)      = plates(p).k;
                component(r) = c;
                medianCol(r) = plates(p).componentMedians(c);
                weightCol(r) = plates(p).componentWeights(c);
                p25Col(r)    = plates(p).componentP25(c);
                p75Col(r)    = plates(p).componentP75(c);
            end
        end
        T = table(entryID, kCol, component, medianCol, weightCol, p25Col, p75Col, ...
                  'VariableNames', {'entry_id', 'k', 'component', 'median', ...
                                    'weight', 'p25', 'p75'});
    end

    function writeComparisonTable(T, csvPath)
        % VIEW (TERMINAL): write the table and report the row count.
        writetable(T, csvPath);
        fprintf('Wrote %d component rows to %s\n', height(T), csvPath);
    end
end
