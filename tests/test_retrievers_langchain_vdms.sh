#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
no_proxy=$no_proxy,$ip_address

function build_docker_images() {
    cd $WORKPATH
    hf_token="dummy"
    docker build --no-cache -t opea/retriever-vdms:comps \
        --build-arg https_proxy=$https_proxy \
        --build-arg http_proxy=$http_proxy \
        --build-arg huggingfacehub_api_token=$hf_token\
        -f comps/retrievers/langchain/vdms/docker/Dockerfile .

}

function start_service() {
    #unset http_proxy
    # vdms
    vdms_port=55555
    docker run -d --name test-comps-retriever-vdms-vector-db \
        -p $vdms_port:$vdms_port   intellabs/vdms:latest
    sleep 10s

    # tei endpoint
    tei_endpoint=5008
    model="BAAI/bge-base-en-v1.5"
    docker run -d --name="test-comps-retriever-tei-endpoint" \
        -p $tei_endpoint:80 -v ./data:/data \
        -e HTTPS_PROXY=$https_proxy -e HTTP_PROXY=$https_proxy \
        --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 \
        --model-id $model
    sleep 30s

    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${tei_endpoint}"

    export COLLECTION_NAME="rag-vdms"

    # vdms retriever
    unset http_proxy
    use_clip=0 #set to 1 if openai clip embedding should be used

    docker run -d --name="test-comps-retriever-vdms-server" -p 5009:7000 --ipc=host \
    -e COLLECTION_NAME=$COLLECTION_NAME -e VDMS_HOST=$ip_address \
    -e https_proxy=$https_proxy -e http_proxy=$http_proxy \
    -e VDMS_PORT=$vdms_port -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN \
    -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e USECLIP=$use_clip \
     opea/retriever-vdms:comps
    sleep 3m
}

function validate_microservice() {


    retriever_port=5009
    URL="http://${ip_address}:$retriever_port/v1/retrieval"
    #test_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")

    test_embedding="[0.3212316218862614, 0.05284697028105079, 0.792736615029739, -0.01450667589035648, -0.7358454555705813, -0.5159104761926909, 0.3535153166047822, -0.6465310827905328, -0.3260418169245214, 0.5427377177268364, 0.839674125021304, 0.27459120894125255, -0.9833857616143291, 0.4763752586395751, 0.7048355150785723, 0.4935209825796325, -0.09655411499027178, -0.5739389241976944, 0.34450497876796815, -0.03401327136919208, -0.8247080270670755, -0.9430721851019634, 0.4702688485035773, 0.3872526674852217, -0.13436894777006136, 0.27166203983338266, 0.7724679346611174, 0.49524109590526666, 0.9810730976435518, 0.2143402533230332, 0.35235793217357947, -0.3199320624935764, -0.3535996110405917, 0.1982603781951089, -0.37547349902996063, -0.6148649695355071, 0.388521078627599, 0.7073360849235228, 0.1768845283243352, -0.38289339223361885, 0.36390326284734775, -0.4790146416310761, -0.5412301982310956, 0.33793186533237507, -0.7028178009236765, -0.6850965350085609, -0.519584428926227, 0.07610032557230206, 0.8173990245819258, 0.6620078274633294, 0.9159029345791101, -0.6353085978752564, 0.5816911666251467, -0.03007583916355916, 0.7405029634324471, 0.43720248036100817, -0.8588961125219283, -0.5267610831146254, 0.17242810571201828, -0.5958637989986995, -0.9424146892733949, 0.593549429279222, -0.6516554787902789, -0.5666971591678356, -0.942676397097636, -0.7754876202156127, 0.4981071621118629, 0.3479716647812874, -0.20905562164787628, -0.01239748867059931, -0.39282697259470645, -0.682776727276128, 0.8490471472078613, 0.9407846472878745, 0.38429459825058054, -0.6217288222979798, 0.7017039943902317, 0.2666859825508645, -0.8350624589077213, -0.6844099142855995, 0.7150220289787632, 0.6172753342426756, 0.3411977212235433, -0.6885106120374, -0.9063819220399785, -0.8409372842391187, -0.8297926800281972, -0.7209991962325382, -0.10750064217958677, 0.3293914797165298, -0.7839812511866298, 0.3413595850264284, 0.9251256529601857, -0.7129635996889019, 0.2032168270911272, -0.744174955251268, 0.7691350055313244, -0.20065548721684312, 0.8869269473893813, -0.02043469943990095, 0.6747773545635596, -0.08840723444251264, 0.29835753335664084, -0.06410433319206965, 0.6915278973312651, 0.35470936730145075, -0.8143883316077478, 0.3700125242841532, 0.21752822647915626, -0.8620510146349405, -0.9872766671960136, -0.4418160577207253, -0.22054594310628928, -0.12301077500821433, -0.32532691454130314, -0.13151154223491113, -0.11476973253362455, -0.6347877217496254, -0.7764229239974911, 0.8494414471799672, -0.8096141861298036, -0.126108099532108, -0.3910538453811505, 0.7416491690145808, -0.9147820237179922, -0.09053536925720418, 0.6536341825563443, 0.655602583013402, 0.1757558598054938, -0.2501459855449637, 0.23414048418314914, -0.2944157385030681, 0.9386472406881659, -0.18806566910431344, -0.29109490690006345, -0.06582041104197667, -0.24458043176038613, 0.22893907834264082, -0.6322528508563678, -0.7885667746432836, 0.10383516801892911, 0.25661930212021256, 0.48395546864077654, 0.25074187080653787, 0.7878158493705165, 0.23874513474134984, -0.18963037155323526, 0.6768315857746809, 0.5323731821887652, 0.23324330999046516, -0.738289178845237, 0.8231931441360549, -0.5243106029457096, 0.21804967641989204, 0.3707592922049536, 0.1970890658467559, 0.6290401053696923, -0.6193312718716564, 0.4319818453521995, -0.4373242547587233, -0.20412719166280646, -0.868724458613944, -0.9426457085574942, 0.7688331784589177, 0.8429476319014946, -0.6928872166553237, -0.3089062124196522, -0.4951601658025162, -0.20786350848417157, -0.1834098357401246, 0.6258630377921288, -0.25204085881527294, -0.6433661815891194, 0.24194250996512046, 0.7945180851525879, 0.6730215739979015, 0.45995755232419877, 0.27685945410814927, 0.7529674957244883, -0.4439881981193141, 0.38722277085649703, 0.4225851985441007, 0.5151867308566294, 0.8592936274009735, -0.5577167356519221, -0.22541015002223674, 0.7872403040580904, -0.12895843621078895, 0.5887160803674254, -0.6121486933005933, -0.45190497189987, 0.5882515994898736, -0.20915972333667443, 0.6412544240387859, -0.9812292190679823, 0.23598351448404986, -0.01874477123769469, -0.5571884049798792, -0.21717058226127106, -0.8566428604555374, -0.7698283820683764, -0.7788953845967042, -0.9695043602118194, 0.2531642774513472, 0.24476771264255004, 0.799177428779027, 0.15892099361251932, 0.2675472976400166, 0.7977537791258142, 0.5682082238828539, -0.45861936031507833, 0.976812562932188, 0.7074171102968665, -0.255345769250928, -0.8903371790301657, 0.7704811965386686, 0.7499406836491052, 0.015867022798163433, 0.023343856172087563, -0.8985882333056163, 0.967943518200411, 0.6738003473613683, 0.500027753964835, -0.25086930359627546, 0.8192342987623937, -0.5553572601867272, -0.5869387659256808, 0.8105241617485164, 0.26722188191476604, -0.3958252448602495, -0.5045071968072412, -0.28738102025143886, 0.9466985876572256, 0.7491954841518662, -0.05398806963889902, 0.5602374066760636, -0.7105267600964871, 0.9183176656578995, -0.7484524873628995, -0.9707740622635459, -0.835248467210193, -0.6698976002755301, -0.9157167347077453, 0.8385470752014215, -0.8484323571440642, 0.1488482374866753, 0.3535389435893035, 0.40201643606217297, -0.39307181109310174, -0.651228451786785, 0.9707155460374848, 0.7578035730666239, -0.916880505891617, 0.7976566483403702, 0.4769359186496589, -0.9056872532891009, 0.5018227509242583, 0.06634988131602104, -0.38876676686204537, -0.20473802582321277, 0.5980365889203325, -0.34935300908506206, 0.5873905336860825, -0.8339160527604776, 0.2903116937984762, -0.9254374424169307, 0.6580958452134436, 0.15246698154103022, -0.6646130474515959, 0.8207084174685697, 0.06879769054023499, 0.6856796611464853, 0.7434402148947985, -0.07417300955086725, -0.37981881059511857, 0.7945700979382095, 0.9465476443316254, 0.7045891367557522, -0.21374560717812052, 0.09707043886320443, 0.40542472035097754, -0.21295063208183063, -0.3638798039778244, 0.27259830494730597, -0.9679565648433712, 0.574009198040323, 0.5453104171463734, 0.4226578254247848, 0.8135241112071945, -0.9913587704531821, -0.5117490950168377, 0.31240764840477486, 0.05726091394767008, -0.44352035546239654, 0.973651830312322, -0.30089019754641044, -0.38110683211990515, 0.12746451891554633, -0.44142668003974683, -0.6085743100333996, 0.6897705314589502, 0.9941017194163115, 0.22931154106427631, -0.38393397164902865, -0.487276417971108, 0.9823011016539693, -0.525188403356583, 0.20472304461076174, -0.549309125745228, 0.8391439613819196, -0.29947371410247614, -0.9587993477785177, 0.49169643064876745, -0.8450431739492874, 0.4992908092405386, 0.8214166011949593, 0.3514461197612715, 0.7052749449063302, -0.456428137096097, -0.21613329759075817, -0.4240696515484821, -0.6072280877366947, -0.19019911975234938, 0.03207563995916485, 0.7832264288656379, -0.9848532944591397, 0.2814057130788894, 0.860398099217986, -0.5757789213121853, -0.6403226820347003, 0.6276892831123779, 0.6966115314942829, -0.5964071917752842, 0.44624318175630373, 0.7747997483259705, -0.5274892594576506, -0.00345488047657061, 0.39694784159551255, -0.32018146543784254, 0.7503113292041483, 0.2279567107684024, -0.6993797573511833, 0.07551046336599065, 0.34912828888955083, 0.4590408940147299, 0.25454507513086266, -0.30882522463970363, -0.4080889783776509, -0.3123706885833979, -0.8906352519220135, -0.8139972234039548, -0.08828963608894047, 0.14503312886836617, -0.3714118896544083, 0.3827783378301277, 0.5438460044018558, 0.5097760438462526, 0.15715247575456592, 0.7656929283612122, 0.2920396353744734, 0.2373440190759446, 0.9526910643357105, 0.1250822784239567, 0.8541819063485603, -0.12747895073713877, 0.5735382473541981, -0.5032516001742902, 0.7413632640531032, -0.7276977107465363, 0.843580565716205, 0.7018464054348241, 0.5586022744519274, 0.8087171435922904, -0.21245941454116735, -0.948838383837346, -0.33122336674310726, -0.6044852681843789, 0.9537863293189539, 0.2536799406315282, -0.6165803849255769, 0.7101896753682724, -0.7295247078012181, -0.7614076971639918, -0.26355996174665797, 0.2821572530049805, -0.31435759840484767, 0.4606279529588946, -0.6454718015595133, 0.29204230021467015, -0.9773214517280517, 0.9018006022750058, 0.41864735598581615, -0.6362219585524242, 0.6393270283675747, 0.8775458814947836, -0.8151570635893794, 0.3439568607968999, 0.29709851503999474, -0.757078876496533, 0.5012539900859203, 0.9894088580102554, -0.7830638861580885, -0.2991021462567893, 0.106227593453466, 0.475717480159388, -0.8190837445165258, 0.7235860704831878, 0.7463245164230621, -0.5005231847044065, 0.6040314499611552, 0.6735380082955229, -0.5547291176872893, -0.9090102518914822, 0.13079236830880614, 0.30122136258272514, -0.6417236467561747, 0.2630310905704383, -0.37163926901056077, 0.20821525595060142, 0.058213575984825905, -0.7186424501121726, 0.7186917038077467, 0.20368227867764155, 0.7957158871869667, -0.8553769107478018, 0.8475526085456688, -0.929286319233819, -0.4084410910607217, -0.18451194893213185, -0.2629665470348457, 0.36380699955097695, 0.2762298083541519, 0.8264334555626198, -0.022207373606218495, -0.32224911623004626, -0.18947254078026798, 0.33627343422225175, 0.6906306880901341, -0.5248865356053838, -0.8976978225060646, -0.9198989266658277, -0.9045058048590318, -0.43074279628622225, 0.9599523380525761, 0.16694571818827875, 0.08638717900194992, 0.24369341180939874, -0.29293980835779454, 0.13980998987643733, -0.9103052978285509, 0.9109674748745353, -0.6189652187256851, -0.30507868365416413, -0.4232217216255978, 0.34784431052206877, -0.8235167119697908, 0.1565512568825982, -0.11476153735499195, -0.5476852944817927, -0.9695366885614041, 0.31387227761880165, -0.8460727492314095, 0.5313339961520958, 0.5605009436841186, 0.04504755045556719, -0.10937916620725119, -0.40867992424849797, -0.9148814576758182, 0.41260731002228, 0.6535850987782705, -0.3956136730481463, 0.03633719317271722, -0.26520169024611917, -0.39307279913859916, 0.8389708129910836, -0.10965192030153337, -0.8114479506343715, 0.6624055258346568, -0.12364857684372677, -0.3391386034226034, 0.5064344415363975, 0.4222558794792024, -0.8920802019539475, 0.8403881748708741, -0.5144930020007417, -0.3961429483392995, -0.9112376538340263, 0.5369991550001529, 0.4099994212177125, 0.8971702224538953, -0.07250674251100442, -0.4123232887614461, -0.4122138364547645, 0.30115503935936516, 0.9140832812087094, -0.37996517983025035, 0.45766194212423583, 0.8778668278803266, -0.871373882496363, 0.9061603981794313, -0.4815792838295849, -0.3540250825062252, 0.47058280496548677, 0.6353307464139133, -0.9084299203157564, 0.32569503818833767, -0.5917177728092791, 0.017982667746413883, -0.39657854384311597, 0.30240291420731147, -0.8789617636583977, 0.398601970442066, -0.9537566407528597, -0.7326801366509474, 0.6394091009367926, -0.24018952260048332, -0.4410443985541457, -0.715250103875068, -0.9531170489995859, 0.8907413230296786, -0.6270483513933209, -0.1278281545077713, 0.6205668124687644, -0.5880492136441298, 0.8458960227498347, 0.5156432304509859, -0.41522707199863196, -0.9971627462302537, 0.967570980171752, -0.1258013547750596, -0.3920054384667395, -0.7579953976551077, -0.5047276085442098, -0.742917134758996, 0.307776046578512, 0.33240724082891204, -0.12439712701067074, 0.8297068611891512, 0.9092972699438713, -0.5553533790744807, -0.9327632085647035, 0.4797798607215402, -0.6407284323825371, 0.23503537288803233, 0.7356444783186646, 0.550461677629142, -0.8859356421536595, -0.06157466053719496, 0.2628024780598055, -0.14515603184459613, -0.9382781600128365, -0.9076306357777459, -0.5661586668239169, -0.5778188698610502, -0.343591139945177, -0.9957519288956789, 3.652203366399931e-05, -0.2850434941249338, 0.9450784913510459, -0.7344049612004591, 0.3966551077940945, 0.9820403785569927, 0.7132254472780228, 0.04475455308790677, 0.7149662286904288, 0.30640286803677386, -0.11825818002978239, 0.9475071024012094, -0.4020573255284672, -0.25210492474829316, -0.9864930649895771, -0.3662338670933165, 0.6528806547589174, 0.23157758222346203, -0.5707934304014186, -0.12462852967839688, 0.1912875382350001, 0.9111205883142817, -0.7227638014501978, -0.36537014763125186, -0.37380198030841805, 0.4707867786085871, -0.5824192322860218, -0.47547092650542666, 0.7836345381645189, 0.7843678847969751, 0.6754328587362883, -0.6670404362153401, 0.7372872996570987, -0.8333262364813818, -0.41971949504499273, -0.7600660277081586, 0.22809249636551576, -0.8923092554006928, -0.28910705230462663, 0.17556387278264474, -0.3120642961908995, -0.08857040909612457, 0.9736924099705169, -0.6425732085916924, 0.5667862783362607, -0.45242262118684295, -0.3366537122702131, -0.21042580668493605, -0.969230642055972, -0.6986186588663355, -0.5420629464988849, 0.8012632695329027, 0.10364503122371205, -0.8288649738571241, -0.7488901002163446, -0.2086447971105505, 0.24528530567671103, -0.1194706644737491, -0.4487125509839567, 0.19757079065420702, 0.9701391397770309, 0.6918580324259651, -0.6609864495230626, -0.5767397650124655, 0.13274852903677803, 0.45790899492650117, 0.6156249211932037, -0.5400854790245104, -0.4871335994554471, -0.37124459518957686, -0.9740961061020355, 0.8132186161153883, 0.5432742278375737, -0.7555629992450097, -0.3626273029276168, 0.3273351801156006, 0.2950481130490956, 0.5899713501222568, 0.1290258276325824, 0.14809153246329188, -0.8527458869128903, -0.45135237009997664, -0.78966354981686, -0.9869505409499153, 0.5440922045096472, -0.5065478252374527, 0.8914118613097968, -0.7009799840752231, -0.37720301784400667, -0.1990418958793818, 0.07895118490326825, 0.43246496862820827, 0.06871630683294172, 0.04584623777009278, -0.34229499350310455, 0.9387219959330184, -0.5381844165951264, 0.4794422861285379, 0.8534951958829573, 0.5734335942167272, -0.85412829706822, -0.7352963908032732, -0.12895000820916747, -0.22552570725823173, -0.5976878733463429, -0.32791035485443487, 0.7202059113861725, 0.39099290295132905, 0.30525825694263764, -0.2266469266742548, -0.03379388729241706, -0.5954645444941691, -0.02422270847921526, 0.2367051711225363, 0.0254309367030352, -0.8571941247598263, 0.6036464885617703, 0.780145197998714, -0.18486284139078912, -0.4861368589284454, -0.2789831003703762, -0.695370188724934, 0.20748300875047643, 0.613995882433769, -0.20040817194169125, 0.8373240273873666, 0.6138944053316708, -0.7863205352137852, -0.7823411702718377, 0.79906295867358, -0.5467331800231525, -0.6344655458958364, -0.9818941753091346, 0.5525644258030062, 0.6262889073747209, 0.9963129049354384, -0.6272737000603017, -0.2716262931036606, 0.2096677033434846, -0.6982262682600213, -0.5674210473085657, 0.24902399542030595, -0.5657568018493333, 0.08618618872017958, 0.5489764282591345, -0.8941510222698827, 0.41351613826944567, -0.5112980841262675, 0.4470615015729351, -0.20725162805621333, -0.08479642143543553, -0.1278591923549064, -0.4999896814124227, 0.9888904679503661, -0.048462424602504495, -0.7019088972627803, 0.24200967459107448, -0.07080934919496995, -0.7205222066189325, 0.8569714457890816, -0.16535406501060956, -0.6995151061411666, -0.002471197183836038, 0.36657456718336245, -0.21418945415378254, 0.8960422717208372, -0.8112144998402944, 0.3367368342692487, -0.1409734233274329, 0.9270438056838188, 0.6449085435355675, -0.42063510394970094, -0.5514753035609532, -0.7824719546926855, 0.27064161179409774, 0.7610801292513893, 0.041332375564573365, -0.4938906089444197, 0.6565606828711339, -0.8175201877660032, -0.7145428710506601, 0.5266689558422335, -0.36373337569732045, -0.4295940430516798, 0.6614123405581125, -0.5795867768963181, 0.09683447902632913, -0.7233160622088481, -0.035259383881968365, 0.44407987368431834, 0.5080824859277744, -0.025605597564321236, -0.33746311986945, 0.8643101724003239, -0.6590382567793307, 0.11251953056040387, -0.5283365207737802, 0.8881578952123139, -0.9796498715072419, -0.8206325632112821, -0.5431772730915239, -0.09628735573638458, 0.8509192593020449, 0.6468967965920123, -0.5886852895684587, -0.25974684548008664, 0.4474352123365879, -0.2199845691372495, 0.7554317108927318, 0.9809450136647395, -0.9430090133566618, 0.23635288316941683]"



    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL")

    #echo "HTTP_STATUS = $HTTP_STATUS"

    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ retriever ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/retriever.log)

        if echo "$CONTENT" | grep -q "retrieved_docs"; then
            echo "[ retriever ] Content is as expected."
        else
            echo "[ retriever ] Content does not match the expected result: $CONTENT"
            docker logs test-comps-retriever-vdms-server >> ${LOG_PATH}/retriever.log
            exit 1
        fi
    else
        echo "[ retriever ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs test-comps-retriever-vdms-server >> ${LOG_PATH}/retriever.log
        exit 1
    fi

    docker logs test-comps-retriever-tei-endpoint >> ${LOG_PATH}/tei.log
}

function stop_docker() {
    cid_retrievers=$(docker ps -aq --filter "name=test-comps-retriever-tei-endpoint*")
    if [[ ! -z "$cid_retrievers" ]]; then
        docker stop $cid_retrievers && docker rm $cid_retrievers && sleep 1s
    fi

    cid_vdms=$(docker ps -aq --filter "name=test-comps-retriever-vdms-server")
    if [[ ! -z "$cid_vdms" ]]; then
        docker stop $cid_vdms && docker rm $cid_vdms && sleep 1s
    fi

    cid_vdmsdb=$(docker ps -aq --filter "name=test-comps-retriever-vdms-vector-db")
    if [[ ! -z "$cid_vdmsdb" ]]; then
        docker stop $cid_vdmsdb && docker rm $cid_vdmsdb && sleep 1s
    fi


}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
