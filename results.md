# Evaluation Results

**validation set path in dataset**: `/test/`

**test set path in dataset** : `/hold/`

## Localization Results

<table>
<thead>
    <tr>
        <td>model</td>
        <td>version</td>
        <td>seed</td>
        <td colspan="2">Test Score</td>
        <td colspan="2">Validation Score</td>
    </tr>
    <tr>
        <td colspan="3"> TTA </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
    </tr>
</thead>
<tbody>
    <tr>
        <td rowspan="4">Resnet34Unet</td>
        <td rowspan="4">1</td>
        <td>0</td>
        <td>0.6590</td>
        <td>0.6643</td>
        <td>0.6542</td>
        <td>0.6590</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.6690</td>
        <td>0.6799</td>
        <td>0.6664</td>
        <td>0.6768</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.6839</td>
        <td>0.6903</td>
        <td>0.6812</td>
        <td>0.6858</td>
    </tr>
    <tr>
        <td> mean agg.</td>
        <td>0.6772</td>
        <td>--</td>
        <td>0.6720</td>
        <td>--</td>
    </tr>
    <tr>
        <td rowspan="4">SeResnext50Unet</td>
        <td rowspan="4">tuned</td>
        <td>0</td>
        <td>0.6963</td>
        <td>0.7002</td>
        <td>0.6957</td>
        <td>0.6967</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.7036</td>
        <td>0.7074</td>
        <td>0.6916</td>
        <td>0.6971</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.7084</td>
        <td>0.7087</td>
        <td>0.6981</td>
        <td>0.7027</td>
    </tr>
    <tr>
        <td>mean agg.</td>
        <td>0.7088</td>
        <td>--</td>
        <td>0.6998</td>
        <td>--</td>
    </tr>
    <tr>
        <td rowspan="4">Dpn92Unet</td>
        <td rowspan="4">tuned</td>
        <td>0</td>
        <td>0.6796</td>
        <td>0.6849</td>
        <td>0.6776</td>
        <td>0.6830</td>
    </tr>
     <tr>
        <td>1</td>
        <td>0.6297</td>
        <td>0.6335</td>
        <td>0.6335</td>
        <td>0.6322</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.6708</td>
        <td>0.6722</td>
        <td>0.6662</td>
        <td>0.6714</td>
    </tr>
    <tr>
        <td>mean agg.</td>
        <td>0.6597</td>
        <td>--</td>
        <td>0.6637</td>
        <td>--</td>
    </tr>
    <tr>
        <td rowspan="4">SeNet154Unet</td>
        <td rowspan="4">1</td>
        <td>0</td>
        <td>0.7348</td>
        <td>0.7393</td>
        <td>0.7261</td>
        <td>0.7302</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.7253</td>
        <td>0.7319</td>
        <td>0.7100</td>
        <td>0.7163</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.7326</td>
        <td>0.7360</td>
        <td>0.7217</td>
        <td>0.7252</td>
    </tr>
    <tr>
        <td>mean agg.</td>
        <td>0.7409</td>
        <td>--</td>
        <td>0.7264</td>
        <td>--</td>
    </tr>
    <tr>
        <td rowspan="8">EfficientUnetB0</td>
        <td rowspan="3">Regular</td>
        <td>0</td>
        <td>0.7692</td>
        <td>0.7739</td>
        <td>0.7634</td>
        <td>0.7666</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.7685</td>
        <td>0.7723</td>
        <td>0.7638</td>
        <td>0.7662</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.7704</td>
        <td>0.7740</td>
        <td>0.7625</td>
        <td>0.7666</td>
    </tr>
    <tr>
        <td rowspan="3">SCSE</td>
        <td>0</td>
        <td>0.7723</td>
        <td>0.7749</td>
        <td>0.7644</td>
        <td>0.7674</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.7707</td>
        <td>0.7737</td>
        <td>0.7628</td>
        <td>0.7682</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.7721</td>
        <td>0.7765</td>
        <td>0.7647</td>
        <td>0.7711</td>
    </tr>
    <tr>
        <td rowspan="2">Wide-SE</td>
        <td>0</td>
        <td>0.7719</td>
        <td>0.7758</td>
        <td>0.7662</td>
        <td>0.7700</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.7754</td>
        <td>0.7754</td>
        <td>0.7664</td>
        <td>0.7682</td>
    </tr>
    <tr>
        <td rowspan="2">EfficientUnetB4</td>
        <td>Regular</td>
        <td>--</td>
        <td>0.7755</td>
        <td>0.7797</td>
        <td>0.7702</td>
        <td>0.7724</td>
    </tr>
    <tr>
        <td>SCSE</td>
        <td>--</td>
        <td>0.7811</td>
        <td>0.7826</td>
        <td>0.7718</td>
        <td>0.7743</td>
    </tr>
    <tr>
        <td rowspan="3">SegFormerB0</td>
        <td rowspan="3">512*512_ade</td>
        <td>0</td>
        <td>0.7602</td>
        <td>0.7281</td>
        <td>0.7543</td>
        <td>0.7214</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.7569</td>
        <td>0.7223</td>
        <td>0.7533</td>
        <td>0.7189</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.7605</td>
        <td>0.7301</td>
        <td>0.7545</td>
        <td>0.7250</td>
    </tr>
</tbody>
</table>

### Meta-Learning

test tasks: `mexico-earthquake`,`joplin-tornado`

<table>
    <thead>
        <tr>
            <td rowspan="2">model</td>
            <td colspan="2">#tasks</td>
            <td rowspan="2">algorithm</td>
            <td colspan="2">meta-optimizer</td>
            <td colspan="2">inner-optimizer</td>
            <td colspan="2">shots</td>
            <td rowspan="2">localization score</td>
        </tr>
        <tr>
            <td>train</td>
            <td>test</td>
            <td>type</td>
            <td>lr</td>
            <td>type</td>
            <td>lr</td>
            <td>support</td>
            <td>query</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">EfficientUnetB0</td>
            <td rowspan="2">17</td>
            <td rowspan="2">2</td>
            <td rowspan="2">MAML</td>
            <td rowspan="2">AdamW</td>
            <td>15e-6</td>
            <td rowspan="2">SGD</td>
            <td>1e-4</td>
            <td>1</td>
            <td>2</td>
            <td>0.5372</td>
        </tr>
        <tr>
            <td>15e-6</td>
            <td>1e-3</td>
            <td>5</td>
            <td>10</td>
            <td>0.4351</td>
        </tr>
    </tbody>
</table>

## Classification Results

<table>
<thead>
    <tr>
        <td>model</td>
        <td>version</td>
        <td>seed</td>
        <td colspan="2">Test Score</td>
        <td colspan="2">Validation Score</td>
    </tr>
    <tr>
        <td colspan="3"> TTA </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
    </tr>
</thead>
<tbody>
    <tr>
        <td rowspan="4">Resnet34Unet</td>
        <td rowspan="4">tuned</td>
        <td>0</td>
        <td>0.1090</td>
        <td>0.0806</td>
        <td>0.1119</td>
        <td>0.0831</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.1466</td>
        <td>0.1174</td>
        <td>0.1264</td>
        <td>0.0997</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.1314</td>
        <td>0.1101</td>
        <td>0.1324</td>
        <td>0.1082</td>
    </tr>
    <tr>
        <td> mean agg.</td>
        <td>0.0860</td>
        <td>--</td>
        <td>0.0832</td>
        <td>--</td>
    </tr>
    <tr>
        <td rowspan="4">SeResnext50Unet</td>
        <td rowspan="4">tuned</td>
        <td>0</td>
        <td>0.6164</td>
        <td>0.6152</td>
        <td>0.6397</td>
        <td>0.6347</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.6135</td>
        <td>0.6069</td>
        <td>0.6012</td>
        <td>0.5991</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.6319</td>
        <td>0.6422</td>
        <td>0.6271</td>
        <td>0.6361</td>
    </tr>
    <tr>
        <td>mean agg.</td>
        <td>0.6360</td>
        <td>--</td>
        <td>0.6301</td>
        <td>--</td>
    </tr>
    <tr>
        <td rowspan="4">Dpn92Unet</td>
        <td rowspan="4">tuned</td>
        <td>0</td>
        <td>0.6564</td>
        <td>0.6657</td>
        <td>0.6387</td>
        <td>0.6441</td>
    </tr>
     <tr>
        <td>1</td>
        <td>0.6233</td>
        <td>0.6343</td>
        <td>0.5869</td>
        <td>0.5813</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.6246</td>
        <td>0.6252</td>
        <td>0.6075</td>
        <td>0.6138</td>
    </tr>
    <tr>
        <td>mean agg.</td>
        <td>0.6460</td>
        <td>--</td>
        <td>0.6258</td>
        <td>--</td>
    </tr>
    <tr>
        <td rowspan="4">SeNet154Unet</td>
        <td rowspan="4">tuned</td>
        <td>0</td>
        <td>0.6916</td>
        <td>0.7034</td>
        <td>0.6684</td>
        <td>0.6722</td>
    </tr>
     <tr>
        <td>1</td>
        <td>0.6216</td>
        <td>0.6342</td>
        <td>0.5889</td>
        <td>0.6123</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.6868</td>
        <td>0.6949</td>
        <td>0.6520</td>
        <td>0.6479</td>
    </tr>
    <tr>
        <td>mean agg.</td>
        <td>0.6954</td>
        <td>--</td>
        <td>0.6596</td>
        <td>--</td>
    </tr>
</tbody>
</table>


